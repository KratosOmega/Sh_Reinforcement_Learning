import win32com.client as com
import numpy as np
import random
import gc
import utils

class Vissim:


    def __init__(self):
        self.vissim = com.dynamic.Dispatch('Vissim.Vissim.1100')
        self.NetworkPath = "C:\\Users\\UAV_MASTER\\Desktop\\_workspace\\Sh_Reinforcement_Learning\\py\\vissim_data\\SH.inpx"
        self.LayoutPath = "C:\\Users\\UAV_MASTER\\Desktop\\_workspace\\Sh_Reinforcement_Learning\\py\\vissim_data\\SH.layx"
        self.SimPeriod = 99999
        self.SimRes = 5
        self.RandSeed = 54
        self.DataCollectionInterval = 60
        self.volume = 5000 #5000
        # -----------------------------------------------------------------------------------
        self.vissim.LoadNet(self.NetworkPath)
        self.vissim.LoadLayout(self.LayoutPath)
        self.vissim.SuspendUpdateGUI()
        self.set_simulation_atts(self.SimPeriod, self.SimRes, self.RandSeed)
        self.set_evaluation_atts(self.SimPeriod, self.DataCollectionInterval)
        self.set_vehicle_input(self.volume)
        self.set_w99cc1distr(103)
        self.vissim.ResumeUpdateGUI()
        # 6 state = flowrate_1, lane_percent_1, lane_percent_2, lane_percent_3, density_1, density_2, density_3
        self.state_space = np.ndarray(shape=(7,), dtype=float)
        # 3 action = speed_limit_1, speed_limit_2, speed_limit_3
        self.action_space = np.ndarray(shape=(3,), dtype=float)
        self.reward_threshold = 0.95 # desired max discharging rate
        self.input_flow_rate = 0 # keep track current upstream flow rate

    def set_w99cc1distr(self, value):
        # value = distance between 2 car (front to back)
        print("===============================")
        print(self.vissim.Net.DrivingBehaviors)
        print("===============================")        
        self.vissim.Net.DrivingBehaviors.ItemByKey(3).SetAttValue("W99cc1Distr", value)

    def set_vehicle_input(self, volume):
        # volumne = # of car per hour in the simulation
        self.vissim.Net.VehicleInputs.ItemByKey(1).SetAttValue("Volume(1)", volume)

    def set_simulation_atts(self, simPeriod, simRes, randSeed):
        self.vissim.Simulation.SetAttValue("simPeriod", simPeriod)
        self.vissim.Simulation.SetAttValue("simRes", simRes)
        self.vissim.Simulation.SetAttValue("randSeed", randSeed) # car stream behavior (how fast car into, how many cars into)
        self.vissim.Simulation.SetAttValue("NumCores", 1)
        self.vissim.Simulation.SetAttValue("UseMaxSimSpeed", True)
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1) # Quick Mode (no car visualization)
        #self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 0) # Normal Mode (car visualization)

    def set_evaluation_atts(self, simPeriod, dataCollectionInterval = 30):
        # |-------|-------|-------|-------|-------| ..... | 5000
        # 0       30      30      30    ...
        self.vissim.Evaluation.SetAttValue("DataCollCollectData", True)
        self.vissim.Evaluation.SetAttValue("DataCollToTime", simPeriod)
        self.vissim.Evaluation.SetAttValue("DataCollFromTime", 0)
        self.vissim.Evaluation.SetAttValue("DataCollInterval", dataCollectionInterval)

        self.vissim.Evaluation.SetAttValue("VehTravTmsCollectData", True)
        self.vissim.Evaluation.SetAttValue("VehTravTmsToTime", simPeriod)
        self.vissim.Evaluation.SetAttValue("VehTravTmsFromTime", 0)
        self.vissim.Evaluation.SetAttValue("VehTravTmsInterval", dataCollectionInterval)

        self.vissim.Evaluation.SetAttValue("VehNetPerfCollectData", True)
        self.vissim.Evaluation.SetAttValue("VehNetPerfToTime", simPeriod)
        self.vissim.Evaluation.SetAttValue("VehRecFromTime", 0)
        self.vissim.Evaluation.SetAttValue("VehNetPerfInterval", dataCollectionInterval)

        self.vissim.Evaluation.SetAttValue("KeepPrevResults", "KeepCurrent")

    def run_single_step(self):
        self.vissim.Simulation.RunSingleStep()

    def run_continuous(self):
        self.vissim.Simulation.RunContinuous()

    def get_simulation_second(self):
        # end is 0  singlesteptime = 1/simres
        return self.vissim.Simulation.SimulationSecond

    def stop_simulation(self):
        self.vissim.Simulation.Stop()

    def set_speed(self, speed_cat, speeds):
        land_id_dict = {
            "speed_input": [1, 2, 3],
            "speed_limit": [4, 5, 6],
        }

        lane_id = land_id_dict[speed_cat]
        spd_nos = self.get_desire_speed_number_array(speeds)

        for i in range(len(lane_id)):
            self.vissim.Net.DesSpeedDecisions.ItemByKey(lane_id[i]).SetAttValue("DesSpeedDistr(10)", spd_nos[i])


    def get_desire_speed_number(self, speed):
        speed_dict = {}
        speed_step = 19
        speed_interval = 5
        speed_init = 30
        speed_vissim_init = 700

        for i in range(speed_step):
            speed_dict[speed_init + i * speed_interval] = speed_vissim_init + i * speed_interval

        return speed_dict[speed]

    def get_desire_speed_number_array(self, speeds):
        length = len(speeds)
        desirespeednums = [0]*length
        for i in range(0, length):
            desirespeednums[i] = self.get_desire_speed_number(speeds[i])
        return desirespeednums

    def run_one_interval(self):
        for i in range(0, self.SimRes * 180):
            self.run_single_step()

    # <editor-fold desc = "region All Vehicle Info"

    def print_links_info(self):
        #Print out Link Info
        for link in self.vissim.Net.Links:
            link_num = link.AttValue["No"]
            link_name = link.AttValue["Name"]
            print("Link %d ( %s )" %(link_name , link_num))

    def print_all_vehicles_info(self): 
        #Print out Vehicle Input Info
        for vehicleInput in self.vissim.Net.VehicleInputs:
            vehicle_input_num = vehicleInput.AttValue["No"]
            vehicle_input_link = vehicleInput.AttValue["Link"]
            print("Vehicle No: %d  VehicleInputLink: %d" %(vehicle_input_num ,vehicle_input_link ))
    
    def print_vehicles_num(self):
        vehicle_nums = self.vissim.Net.VehicleInputs.GetMultiAttValues("No")
        for num in vehicle_nums:
            print("Vehicle No : %d" + num)

    def get_all_vehicles_by_id(self):
        return ConsoleApplication1vissim.Net.Vehicles.GetMultiAttValues("No")

    def get_all_vehicles_by_type(self):
        return self.vissim.Net.Vehicles.GetMultiAttValues("VehType")

    def get_all_vehicles_by_lane(self):
        return self.vissim.Net.Vehicles.GetMultiAttValues("Lane")

    def get_all_vehicles_by_link(self):
        return self.vissim.Net.Vehicles.GetMultiAttValues("Link")

    def get_all_vehicles_by_pos(self):
        return self.vissim.Net.Vehicles.GetMultiAttValues("Pos")

    def get_all_vehicles_by_lanes(self, link_id,  lane_id):
        """
        3-1, 3-2, 3-3
        """
        lane_vehs_num_obj = self.vissim.Net.Links.ItemByKey(link_id).Lanes.ItemByKey(lane_id).Vehs.GetMultiAttValues("No")
        return len(lane_vehs_num_obj)

    # </editor-fold>


    # <editor-fold desc = "LinkInfo"

    def get_link_ids(self):
        return self.vissim.Net.Links.GetMultiAttValues("No")

    def get_link_total_lanes(self):
        return self.vissim.Net.Links.GetMultiAttValues("NUMLANES")

    def get_link_vehs_by_num(self, lkn):
        return self.vissim.Net.Links.ItemByKey(lkn).Vehs.GetMultiAttValues("No")

    def get_link_vehs_by_type(self, lkn):
        return self.vissim.Net.Links.ItemByKey(lkn).Vehs.GetMultiAttValues("VehType")
    
    def get_lanes_by_linkid(self, linkid):
        return self.vissim.Net.Links.ItemByKey(linkid).AttValue("NumLanes")


    # </editor-fold>

    # <editor-fold desc = "Data Collection Result"

    def get_current_data_collection_result_vehs(self, data_collection_group_id):
        data_collection_measurement = self.vissim.Net.DataCollectionMeasurements
        return data_collection_measurement.ItemByKey(data_collection_group_id).AttValue("Vehs(Current, Last, All)")

    def get_current_data_collection_result_speedavgarith(self, data_collection_group_id):
        data_collection_measurement = self.vissim.Net.DataCollectionMeasurements
        return Convert.ToDouble(data_collection_measurement.ItemByKey(data_collection_group_id).AttValue("SpeedAvgArith(Current, Last, All)"))

    def get_current_data_collection_result_travtm(self, data_collection_group_id):
        data_collection_measurement = self.vissim.Net.DataCollectionMeasurements
        return data_collection_measurement.ItemByKey(data_collection_group_id).AttValue("Vehs(Current, Last, All)")

    def get_data_collection_result_vehs(self, data_collection_group_id, timeInterval):
        data_collection_measurement = self.vissim.Net.DataCollectionMeasurements
        attribute = "Vehs(Current," + timeInterval + ", All)"
        return data_collection_measurement.ItemByKey(self, data_collection_group_id).AttValue(attribute)

    def get_data_collection_result_speedavgarith(self, data_collection_group_id, timeInterval):
        data_collection_measurement = self.vissim.Net.DataCollectionMeasurements
        attribute = "SpeedAvgArith(Current," + timeInterval + ", All)"
        return data_collection_measurement.ItemByKey(data_collection_group_id).AttValue(attribute)


        # </editor-fold>


# <editor-fold desc = "Vehicle Travel Time Result"
    def get_current_vehicle_travel_time_vehs(self, vttId):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        return traveltimes.ItemByKey(vttId).AttValue("Vehs(Current, Last, All)")

    def get_current_vehicle_travel_time_travtm(self, vttId):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        return traveltimes.ItemByKey(vttId).AttValue("TravTm(Current, Last, All)")
    def get_current_vehicle_travel_time_disttrav(self, vttId):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        return traveltimes.ItemByKey(vttId).AttValue("DistTrav(Current, Last, All)")

    def get_vehicle_travel_time_vehs(self, vttId, timeInterval):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        attribute = "Vehs(Current," + timeInterval + ", All)"
        return traveltimes.ItemByKey(vttId).AttValue(attribute)

    def get_vehicle_travel_time_travtm(self, vttId, timeInterval):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        attribute = "TravTm(Current," + timeInterval + ", All)"
        return traveltimes.ItemByKey(vttId).AttValue(attribute)

    def get_vehicle_travel_time_disttrav(self, vttId, timeInterval):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        attribute = "DistTrav(Current," + timeInterval + ", All)"
        return traveltimes.ItemByKey(vttId).AttValue(attribute)

    def get_vehicle_travel_time_dist(self, vttId):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        return traveltimes.ItemByKey(vttId).AttValue("Dist")

    def get_vehicle_tracel_time_startlink(self, vttId):
        traveltimes = self.vissim.Net.VehicleTravelTimeMeasurements
        return traveltimes.ItemByKey(vttId).AttValue("StartLink")

    def get_current_density(self, vttid):
        start_link = self.get_vehicle_tracel_time_startlink(vttid)
        num_lanes = self.get_lanes_by_linkid(start_link)
        travelTm_vhs = self.get_current_vehicle_travel_time_vehs(vttid)
        distance = self.get_current_vehicle_travel_time_disttrav(vttid)
        time = self.get_current_vehicle_travel_time_travtm(vttid)
        density = self.calc_density(travelTm_vhs, self.DataCollectionInterval, time, distance, num_lanes)
        return density
        # </editor-fold>

    # <edit-fold des = "calculation functions"
        
    def calc_link_desity(self, lnk):
        length = System.Convert.ToDouble(self.vissim.Net.Links.ItemByKey(lnk).AttValue("Length2D"))
        num_lanes = System.Convert.ToInt32(self.vissim.Net.Links.ItemByKey(lnk).AttValue("NumLanes"))
        temp = self.get_link_vehs_by_num(lnk)
        num_vehs = temp.Length / 2
        density = num_vehs / (num_lanes * length / 1600) #  veh/mi/ln
        return round(density, 4)

   
    def calc_flow_rate(self, num_vehs, timeinterval):
        flow_rate = float(num_vehs) * (3600.0 / float(timeinterval)) #  veh/h
        return round(flow_rate, 4)
    
    def calc_density(self, num_vehs, timeinterval, time, distance, num_lane):
        flow_rate = float(num_vehs) * (3600.0 / float(timeinterval)) # veh/h
        velocity = float(distance) / float(time)
        density = flow_rate / (float(num_lane) * velocity)
        return round(density, 4)

        # </editor-fold>

    def get_rand_speed(self, speed_init, speed_interval, step_min, step_max):
        return speed_init + random.randint(step_min, step_max) * speed_interval


    def reset(self, actions=[50, 50, 50], count=1, run_times=(180*5)):
        # set input speed for SH zone
        self.stop_simulation()
        self.set_speed("speed_input", actions)
        flow_rate = 0.0
        density = 0.0

        for i in range(0, run_times):
            self.run_single_step()

        vehs_pass_to_acc = self.get_current_data_collection_result_vehs(1)
        self.input_flow_rate = flow_rate = self.calc_flow_rate(vehs_pass_to_acc, self.DataCollectionInterval)

        density1 = self.get_all_vehicles_by_lanes(3, 1)
        density2 = self.get_all_vehicles_by_lanes(3, 2)
        density3 = self.get_all_vehicles_by_lanes(3, 3)

        # ---------------------------------------------------------- using normalized densities
        acc_length = 1500
        density_all = density1 + density2 + density3

        lane_percent_1 = round(density1 / density_all, 4)
        lane_percent_2 = round(density2 / density_all, 4)
        lane_percent_3 = round(density3 / density_all, 4)
        density1 = round(density1 / acc_length, 4)
        density2 = round(density2 / acc_length, 4)
        density3 = round(density3 / acc_length, 4)
        normalized_flow_rate = round(flow_rate / self.volume, 4)
        # ----------------------------------------------------------

        state = np.array([normalized_flow_rate, lane_percent_1, lane_percent_2, lane_percent_3, density1, density2, density3])

        return state

    def step(self, actions, run_times=(180*5)):
        self.set_speed("speed_input", actions)
        reward = 0

        for i in range(0, run_times):
            self.run_single_step()

        # get reward (discharging rate)
        vehs_pass_to_bn = self.get_current_data_collection_result_vehs(4)
        discharging_rate = self.calc_flow_rate(vehs_pass_to_bn, self.DataCollectionInterval)

        #------------------------------------------------------------------------------------ reward logic blcok
        #"""
        # no shock wave involved logic

        reward = round(discharging_rate / self.volume, 4)

        # ###################################################################################
        
        """
        # shock wave involved logic

        diff = discharging_rate - self.input_flow_rate
        if diff > 0:
            print("-------- shock wave warning, applied penalty!")
            reward = round(discharging_rate / (self.volume * 1.5), 4) # TODO: find a better flow_rate threshold for shock wave
        else:
            reward = round(discharging_rate / self.volume, 4)
        """
        #------------------------------------------------------------------------------------

        vehs_pass_to_acc = self.get_current_data_collection_result_vehs(1)
        flow_rate = self.calc_flow_rate(vehs_pass_to_acc, self.DataCollectionInterval)

        density1 = self.get_all_vehicles_by_lanes(3, 1)
        density2 = self.get_all_vehicles_by_lanes(3, 2)
        density3 = self.get_all_vehicles_by_lanes(3, 3)

        # ---------------------------------------------------------- using normalized densities
        acc_length = 1500
        density_all = density1 + density2 + density3

        lane_percent_1 = round(density1 / density_all, 4)
        lane_percent_2 = round(density2 / density_all, 4)
        lane_percent_3 = round(density3 / density_all, 4)
        density1 = round(density1 / acc_length, 4)
        density2 = round(density2 / acc_length, 4)
        density3 = round(density3 / acc_length, 4)
        normalized_flow_rate = round(flow_rate / self.volume, 4)
        # ----------------------------------------------------------

        # set state (flow rate, density of [SH, Acc])
        state = np.array([normalized_flow_rate, lane_percent_1, lane_percent_2, lane_percent_3, density1, density2, density3])

        # set bottle next discharging rate threshold
        #terminal = reward > self.reward_threshold
        terminal = False

        return state, reward, terminal

    def traffic_no_sh(self, speed=[70, 70, 70], count=1, run_times=(180*5)):
        self.set_speed("speed_input", speed)
        reward = 0

        for i in range(0, run_times):
            self.run_single_step()

        # get reward (discharging rate)
        vehs_pass_to_bn = self.get_current_data_collection_result_vehs(4)
        discharging_rate = self.calc_flow_rate(vehs_pass_to_bn, self.DataCollectionInterval)

        #------------------------------------------------------------------------------------ reward logic blcok

        # no shock wave involved logic

        reward = round(discharging_rate / self.volume, 4)

        # ###################################################################################

        """
        # shock wave involved logic

        diff = discharging_rate - self.input_flow_rate
        if diff > 0:
            print("-------- shock wave warning, applied penalty!")
            reward = round(discharging_rate / (self.volume * 1.5), 4) # TODO: find a better flow_rate threshold for shock wave
        else:
            reward = round(discharging_rate / self.volume, 4)
        """
        #------------------------------------------------------------------------------------

        return reward

    def record_traffic_no_sh(self, max_episodes, max_steps, speed = [70, 70, 70]):
        for episode in range(max_episodes):
            print("---------------------------- " + str(episode))
            cumulative_r = 0

            try:
                for t in range(0, max_steps):
                    reward = self.traffic_no_sh(speed)
                    cumulative_r += reward
                    gc.collect()

                avg_r = cumulative_r / max_steps
                utils.updateReport(r"\report\traffic_no_sh_report.csv", [avg_r])
            except:
                print("----------------------- oops...")


# ###############################################################################
# Global method
# ###############################################################################

def test_sars():
    vissim = Vissim()

    state = vissim.reset([
        vissim.get_rand_speed(30, 5, 0, 18), 
        vissim.get_rand_speed(30, 5, 0, 18), 
        vissim.get_rand_speed(30, 5, 0, 18)
    ])

    while True:
        state, reward, terminal = vissim.step([
            vissim.get_rand_speed(30, 5, 0, 18), 
            vissim.get_rand_speed(30, 5, 0, 18), 
            vissim.get_rand_speed(30, 5, 0, 18)
            ])

        print("----------------------------")
        print(state)
        print(reward)
        print("----------------------------")

def collect_traffic_data():
    max_episodes = 10000
    max_steps = 200
    speed = [75, 75, 75]

    vissim = Vissim()
    vissim.record_traffic_no_sh(max_episodes, max_steps, speed)


#"""
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # ------------------------ testing sars data flow
    #test_sars()
    # ------------------------ collect traffic discharging rate w/o SH
    collect_traffic_data()
#"""
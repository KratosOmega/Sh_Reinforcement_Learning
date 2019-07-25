import win32com.client as com
import numpy as np

class Vissim:
    vissim = com.dynamic.Dispatch('Vissim.Vissim.1100')
    NetworkPath = "C:\\Users\\UAV_MASTER\\Desktop\\vissim_data\\SH.inpx"
    LayoutPath = "C:\\Users\\UAV_MASTER\\Desktop\\vissim_data\\SH.layx"
    #NetworkPath = "D:\School11111111111111111111111111111\Coop2019\Summer\Vissim 2\SH.inpx"
    #LayoutPath = "D:\School11111111111111111111111111111\Coop2019\Summer\Vissim 2\SH.layx"
    SimPeriod = 99999
    SimRes = 5
    RandSeed = 54
    DataCollectionInterval = 60
    volume = 5000

    def __init__(self):
        self.vissim.LoadNet(self.NetworkPath)
        self.vissim.LoadLayout(self.LayoutPath)
        self.vissim.SuspendUpdateGUI()
        self.set_simulation_atts(self.SimPeriod, self.SimRes, self.RandSeed)
        self.set_evaluation_atts(self.SimPeriod, self.DataCollectionInterval)
        self.set_vehicle_input(self.volume)
        self.set_w99cc1distr(103)
        self.vissim.ResumeUpdateGUI()
        # 6 state = flowrate_1, density_1, flowrate_2, density_2, flowrate_3, density_3
        self.state_space = np.ndarray(shape=(6,), dtype=float)
        # 3 action = speed_limit_1, speed_limit_2, speed_limit_3
        self.action_space = np.ndarray(shape=(3,), dtype=float)


    def set_w99cc1distr(self, value):
        self.vissim.Net.DrivingBehaviors.ItemByKey(3).SetAttValue("W99cc1Distr", value)

    def set_vehicle_input(self, volume):
        self.vissim.Net.VehicleInputs.ItemByKey(1).SetAttValue("Volume(1)", volume)

    def set_simulation_atts(self, simPeriod, simRes, randSeed):
        self.vissim.Simulation.SetAttValue("simPeriod", simPeriod)
        self.vissim.Simulation.SetAttValue("simRes", simRes)
        self.vissim.Simulation.SetAttValue("randSeed", randSeed)
        self.vissim.Simulation.SetAttValue("NumCores", 1)
        self.vissim.Simulation.SetAttValue("UseMaxSimSpeed", True)
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1) #Disable the visibility of all dynamic elements

    def set_evaluation_atts(self, simPeriod, dataCollectionInterval = 30):
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

    def set_all_desire_speed(self, speeds):
        count = self.vissim.Net.DesSpeedDecisions.Count
        spd_nos = self.get_desire_speed_number_array(speeds)
        i = 0
        while i <= count - 3:
            index = i // 3
            self.vissim.Net.DesSpeedDecisions.ItemByKey(i + 1).SetAttValue("DesSpeedDistr(10)", spd_nos[index])
            self.vissim.Net.DesSpeedDecisions.ItemByKey(i + 1).SetAttValue("DesSpeedDistr(70)", spd_nos[index])
            self.vissim.Net.DesSpeedDecisions.ItemByKey(i + 2).SetAttValue("DesSpeedDistr(10)", spd_nos[index])
            self.vissim.Net.DesSpeedDecisions.ItemByKey(i + 2).SetAttValue("DesSpeedDistr(70)", spd_nos[index])
            self.vissim.Net.DesSpeedDecisions.ItemByKey(i + 3).SetAttValue("DesSpeedDistr(10)", spd_nos[index])
            self.vissim.Net.DesSpeedDecisions.ItemByKey(i + 3).SetAttValue("DesSpeedDistr(70)", spd_nos[index])
            i += 3

    def get_desire_speed_number(self, speed):
        speed_dict = {
            30: 700,
            35: 705,
            40: 710,
            45: 715,
            50: 720,
            55: 725,
            60: 730,
            65: 735,
            70: 740,
            75: 745,
            80: 750,
            85: 755,
            90: 760,
            95: 765,
            100: 770,
            105: 775,
            110: 780,
            115: 785,
            120: 790,
        }
        return speed_dict[speed]

    def get_desire_speed_number_array(self, speeds):
        length = len(speeds)
        desirespeednums = [0]*length
        for i in range(0, length):
            desirespeednums[i] = self.get_desire_speed_number(speeds[i])
        return desirespeednums

    def reset(self, count=1, actions=[45, 55, 60, 65, 70, 75, 80], run_times=(180*5)):
        self.set_all_desire_speed(actions)
        flow_rate = 0.0
        density = 0.0

        for i in range(0, run_times):
            self.run_single_step()

        datapo1_vehs = self.get_current_data_collection_result_vehs(1)
        flow_rate = self.calc_flow_rate(datapo1_vehs, self.DataCollectionInterval)

        density1 = self.get_current_density(5)
        density2 = self.get_current_density(6)
        density3 = self.get_current_density(7)
        density4 = self.get_current_density(8)
        density5 = self.get_current_density(9)
        density6 = self.get_current_density(10)

        state = np.array([flow_rate, density1, density2, density3, density4, density5, density6])

        return state

    def step(self, actions, run_times=(180*5)):
        self.set_all_desire_speed(actions)
        Reward = 0

        for i in range(0, run_times):
            self.run_single_step()

        # get reward (discharging rate)
        datapo4_vehs = self.get_current_data_collection_result_vehs(4)
        reward = self.calc_flow_rate(datapo4_vehs, self.DataCollectionInterval)
        
        datapo1_vehs = self.get_current_data_collection_result_vehs(1)
        flow_rate = self.calc_flow_rate(datapo1_vehs, self.DataCollectionInterval)

        density1 = self.get_current_density(5)
        density2 = self.get_current_density(6)
        density3 = self.get_current_density(7)
        density4 = self.get_current_density(8)
        density5 = self.get_current_density(9)
        density6 = self.get_current_density(10)

        # set state (flow rate, density of [SH, Acc])
        state = np.array([flow_rate, density1, density2, density3, density4, density5, density6])

        # set bottle next discharging rate threshold
        terminal = reward > 6000

        return state, reward, terminal


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
        return self.vissim.Net.Links.ItemByKey(link_id).Lanes.ItemByKey(lane_id).Vehs

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
        return round(density, 2)

   
    def calc_flow_rate(self, num_vehs, timeinterval):
        flow_rate = float(num_vehs) * (3600.0 / float(timeinterval)) #  veh/h
        return round(flow_rate, 2)
    
    def calc_density(self, num_vehs, timeinterval, time, distance, num_lane):
        flow_rate = float(num_vehs) * (3600.0 / float(timeinterval)) # veh/h
        velocity = float(distance) / float(time)
        density = flow_rate / (float(num_lane) * velocity)
        return round(density, 2)

        # </editor-fold>


# test:
if __name__ == '__main__':
    vissim = Vissim()
    vissim.reset()

import datetime
import dateutil.tz
import os
import csv

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')

def updateReport(file_name, entry):
	path = os.getcwd() + file_name
	with open(path, 'a', newline='') as fileWriter:
		writer = csv.writer(fileWriter)
		writer.writerow(entry)
	fileWriter.close()
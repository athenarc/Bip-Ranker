#!/usr/bin/python

# Implementation of Time-Aware Ranking for Spark using DataFrames

# Program proceeds as follows:
# We create an dataframe of schema ["paper", "score"]
# and another of the schema ["paper", "cited_paper_list"].

# TODO: Use this for python 2
from __future__ import print_function
# ---------- Imports ------------- #
import sys
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
# Functions to effectively handle data 
# manipulation for DataFrames
import pyspark.sql.functions as F
# Diagnostics
import time
from timeit import default_timer as timer
from datetime import timedelta
from pyspark.sql.types import IntegerType
# -------------------------------- #

# Print output message if not run correctly
###########################################
if len(sys.argv) < 7:
	print ('\nUsage: ./timeawarerank.py <input_file> <gamma> <current_year> <mode:RAM|ECM> <num_partitions> <checkpoint_dir> <optional: checkpoint_mode> <optional:alpha> <optional: max_error> \n')
	sys.exit(0)
###########################################

############################################
# Read input parameters
input_file 	= sys.argv[1]
gamma		= float(sys.argv[2])
current_year	= int(sys.argv[3])
tar_type	= sys.argv[4].lower()
# Set default partitions to 1
try:
	num_partitions = int(sys.argv[5])
except:
	num_partitions = 1
	print ("For num partitions I found: " + str(sys.argv[5]))

try:
	checkpoint_dir = sys.argv[6]
except:
	checkpoint_dir = "check"
	
# If it's not specified whether to use RAM or ECM, we default to RAM
if tar_type not in ['ram','ecm']:
	tar_type = 'ram'
	
# Set checkpoint mode / default checkpoint mode
try:
        checkpoint_mode = sys.argv[7]
except:
        checkpoint_mode = 'dfs'		
		
# Read ECM parameter
if tar_type != 'ram':
	try:
		alpha 		= float(sys.argv[8])
		max_error  	= float(sys.argv[9])
	except:
		print ('\nUsage: ./timeawarerank.py <input_file> <gamma> <current_year> <mode:RAM|ECM> <num_partitions> <checkpoint_dir> <optional: checkpoint_mode> <optional:alpha> <optional: max_error> \n')
		sys.exit(0)
# Set the mode by default as local. 
# If data is read from hdfs we switch to cluster
mode = 'local'
if input_file.startswith('hdfs://'):
	mode = 'distributed'
	
# Set the mode by default as dfs. 
if checkpoint_mode not in ['dfs', 'local']:
	checkpoint_mode = 'dfs'	
############################################
# Calculate total running time
program_start_time = check_time = timer()
initialisation_time = time.time()
# Initialize spark context and spark-specific params
#####################################################################################################
if  mode == 'local':
	conf = SparkConf().setMaster('local').setAppName('Spark Time-Aware Ranking')
else:
	conf = SparkConf().setAppName('Spark Time-Aware Ranking')
# Pass configuration to spark context
sc = SparkContext(conf = conf)
# Session is needed for spark.sql library, in order to use DataFrames
if mode == 'local':
	sp_session = SparkSession.builder.master('local').appName('Spark Time-Aware Ranking').getOrCreate()
else:
	sp_session = SparkSession.builder.config('spark.cleaner.referenceTracking.cleanCheckpoints', 'true').appName('Spark Time-Aware Ranking').getOrCreate()
# Turn off logging (possible values: OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL
sc.setLogLevel('OFF')

# Set checkpoint dir
sc.setCheckpointDir(checkpoint_dir)
#####################################################################################################
#################################
# Get hdfs file handler object. 
# This solution, although not elegant, does not rely on third party libaries
# Source: https://diogoalexandrefranco.github.io/interacting-with-hdfs-from-pyspark/
URI = sc._gateway.jvm.java.net.URI
Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
# Get master prefix from input file path
master_prefix = "/".join(input_file.split("/")[:5])
fs = FileSystem.get(URI(master_prefix), sc._jsc.hadoopConfiguration())
#################
# Testing
# print ("Master prefix is: " + master_prefix) 
# for remote_file in fs.listStatus( Path( "/".join(input_file.split("/")[:-1])) ):
# 	print( remote_file )
# sys.exit(0)
#################

#################################
# Set CiteRank-Specific params  #
#------------------------------ #
# Set initial error to high value
error 	   = 1000
# Read the initial input
if mode == 'local':
	input_data = sc.textFile('file:' + input_file)
elif mode == 'distributed':
	print("\n\nReading input from hdfs\n\n")
	try:
		input_data = sc.textFile(input_file)
	except:
		input_data = sc.textFile(input_file + "/*")
# Get number of nodes (one node-record per line)
num_nodes = float(input_data.count())
# Count iterations
iterations = 1
##################################
# Display CR parameterisation
print ("\n# ------------------------------------ #")
print ("Input: " + input_file)
print ("Gamma: " + str(gamma))
print ("Current Year: " + str(current_year))
print ("Ranking mode: " + tar_type.upper())
if tar_type == 'ecm':
	print ("Alpha: " + str(alpha))
	print ("Convergence Error: " + str(max_error))
print ("Number of nodes: " + str(num_nodes))
print ("Number of partitions: " + str(num_partitions) )
print ("Checkpoint mode: " + str(checkpoint_mode))
print ("Checkpoint dir: " + checkpoint_dir)
print ("# ------------------------------------ #\n")

print ("Initialising", end='')
sys.stdout.flush()
##################################################
# Initialise SPARK Data

# Get a DataFrame with pairs of <node, year>.
# Recall the input format to understand what the following map function does:
# <paper> <tab> <cited_papers|num_cited_papers|score> <tab> <previous_score> <tab> <publication_year>
paper_years = input_data.map(lambda x: (x.split()[0].strip(), x.split()[-1].strip())).toDF(['paper','year'])\
			.withColumn('year_fixed', F.when( (F.col('year').cast(IntegerType()) < 1000) | (F.col('year').cast(IntegerType()) > int(current_year)) | (F.col('year') == "\\N"), 0).otherwise(F.col('year')))
# We have a case of erroneous years when translating openaire IDs. 
# We insert a year validation here (has to be in 1000-2021)
#################
# TESTING
# paper_years.groupBy("year_fixed").count().sort(F.col('count').asc()).show(4000, False)
# sys.exit(0)
#################
paper_years = paper_years.select('paper', F.col('year_fixed').alias('year')).cache()

# Get a DataFrame with pairs of <node, cited_list> (see comment above for input format) 
# Keep only papers that DO cite other papers
outlinks = input_data.map(lambda x: (x.split()[0].strip(), x.split()[1].rsplit("|",2)[0].split(",")))\
		     .toDF(['paper', 'cited_papers'])\
		     .filter(F.col('cited_papers')[0] != "0")\
		     .repartition(num_partitions, 'paper').cache()

print(".", end = '')
sys.stdout.flush()

# If we do ECM calculations, we need to additionally initialise
# TODO: write this
if tar_type == 'ecm':
	scores 		= input_data.map(lambda x: (x.split()[0].strip(),0, x.split()[-1].strip()) ).toDF(['paper', 'score', 'year']).repartition(num_partitions, 'paper').cache()
	previous_scores = scores.select('paper', F.col('score').alias('previous_score'), 'year')
	
###########################################################
# Continue intialisation message
print(". Took: %s seconds!" % (time.time()-initialisation_time))
print("Starting!")
###########################################################

# IF we use RAM, only one calculation is required....
if tar_type == 'ram':
	scores = outlinks.join(paper_years, 'paper')\
			 .select(F.explode(outlinks.cited_papers).alias('paper'), (gamma ** (current_year - F.col('year')) ).alias('transferred_score'))\
			 .groupBy('paper')\
			 .agg(F.sum('transferred_score').alias('score'))\
			 .join(paper_years, 'paper', 'right_outer')\
			 .fillna(0.0, ['score'])\
			 .select('paper', 'score')

# .... Otherwise we have to iteratively calculate scores
else:

	# Calculate the initial score - since in each iteration the we must use the previous score we do the initial step separately
	scores = outlinks.join(scores, 'paper')\
			 .select(F.explode(outlinks.cited_papers).alias('paper'), (alpha*(gamma ** (current_year - F.col('year')))).alias('transferred_score'), 'year')\
			 .groupBy('paper')\
			 .agg(F.sum('transferred_score').alias('score'))\
			 .join(previous_scores, 'paper', 'right_outer')\
			 .select('paper','score','year').repartition(num_partitions, 'paper')\
			 .fillna(0.0, ['score']).cache()

	previous_scores = scores.select('paper', F.col('score').alias('previous_score'), 'year')
	# Loop until error is less than the threshold given
	while error >= max_error:

		#######################################################
		# Time the iteration
		iteration_time = time.time()
		# Notify user
		print ("\n# ------------------------------------ #\n")
		#######################################################

		# .select('paper', (F.col('score') * (alpha ** (iterations)) ).alias('running_ecm'))\
		# ---------------------------------------------------------------------- #
		# 1. Do the basic map-reduce step to calculate scores
		scores = outlinks.join(scores, 'paper')\
				 .select(F.explode(outlinks.cited_papers).alias('paper'), (alpha*(gamma ** (current_year - F.col('year')))*F.col('score')).alias('transferred_score'))\
				 .groupBy('paper')\
				 .agg(F.sum('transferred_score').alias('running_ecm'))\
				 .join(previous_scores, 'paper', 'right_outer')\
				 .repartition(num_partitions, 'paper')\
				 .fillna(0.0, 'running_ecm')\
				 .select('paper', 'running_ecm', 'previous_score', (F.col('running_ecm')+F.col('previous_score')).alias('ecm'), 'year')

		# Keep a local checkpoint of the scores since they will be used again in the next iteration
		if checkpoint_mode == 'dfs':
			# Update checkpoint directory
			if checkpoint_dir.endswith("/"):
				check_dir = checkpoint_dir[:-1]
			else:
				check_dir = checkpoint_dir
			sc.setCheckpointDir(check_dir + "_TAR_ECM_iteration_" +  str(iterations+1))
			# Checkpoint the data
			scores = scores.checkpoint()
			# Clean previous checkpoint directory
			fs.delete( Path(checkpoint_dir + "_TAR_ECM_iteration_" +  str(iterations)) )
		elif checkpoint_mode == 'local':
			scores = scores.localCheckpoint()
		# ---------------------------------------------------------------------- #
		# 2. Calculate error between successive scores -> since scores are incrementally calculated 
		# by adding an extra part in each iteration, we only need to find the maximum added score
		error = scores.agg({'running_ecm':'max'}).collect()[0][0]

		# ---------------------------------------------------------------------- #
		# 3. Re-initialise variables and DataFrames

		# Keep only running citerank part in scores and keep the previous accumulated citerank values as the previous score
		previous_scores = scores.select('paper', F.col('ecm').alias('previous_score'), 'year')
		scores 		= scores.select('paper', F.col('running_ecm').alias('score'), 'year')
		# ---------------------------------------------------------------------- #
		print("--- Iteration " + str(iterations) + " - Time: " + "{:.2f}".format(time.time()-iteration_time) + " (s) - Error: "  + str(error) + " ---")
		# Count iterations
		iterations += 1


print ("\n# ------------------------------------ #\n")
print("Finished! Writing output to file.")

# Write output in tab delimited format in LOCAL file system
if mode == 'local':
	input_file_name = input_file.split("/")[-1]
	# RAM
	if tar_type.lower() == 'ram':
		scores.write.format('csv').mode("overwrite").option("delimiter","\t").save("RAM_" + input_file_name + "_local_c" + str(gamma) + "_year" + str(current_year))
		print("RAM_" + input_file_name + "_local_c" + str(gamma) + "_year" + str(current_year))
	# ECM	
	else:
		previous_scores.write.format('csv').mode("overwrite").option("delimiter","\t").save("ECM_" + input_file_name + "_local_c" + str(gamma) + "_year" + str(current_year) + "_a" + str(alpha) + "_error" + str(max_error))
		print("ECM_" + input_file_name + "_local_c" + str(gamma) + "_year" + str(current_year) + "_a" + str(alpha) + "_error" + str(max_error))
# Output in HDFS
elif mode == 'distributed':
	input_file_name = input_file.split("/")[-1]
	output_prefix = "/".join(input_file.split("/")[3:-1])
	# RAM
	if tar_type.lower() == 'ram':
		print("Writing to: /" + output_prefix + "/TAR_results/RAM_" + input_file_name.replace(".gz", "") + "_c" + str(gamma) + "_year" + str(current_year))
		scores.write.mode("overwrite").option("delimiter","\t").csv("/" + output_prefix + "/RAM_" + input_file_name.replace(".gz", "") + "_c" + str(gamma) + "_year" + str(current_year))
	# ECM	
	else:
		print("Writing to: /" + output_prefix + "/ECM_" + input_file_name.replace(".gz", "") + "_c" + str(gamma) + "_year" + str(current_year) + "_a" + str(alpha) + "_error" + str(max_error))
		previous_scores.write.mode("overwrite").option("delimiter","\t").csv("/" + output_prefix + "/ECM_" + input_file_name.replace(".gz", "") + "_c" + str(gamma) + "_year" + str(current_year)  + "_a" + str(alpha) + "_error" + str(max_error))

program_end_time = timer()
print ("\n\nRanking program took: " + str(timedelta(seconds=program_end_time-program_start_time)))		
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################

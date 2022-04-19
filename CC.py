#!/usr/bin/python

# Implementation of Indicator counting citations up to X
# years after publication - or all - depending on input

# Program proceeds as follows:
# We create an dataframe of schema ["citing_paper", "cited_paper", "citing_paper_year"]
# the first and last columns we read from file, the second is created by a split of the
# cited list. 
# We join this schema with the publication years based on the CITED paper to get a 
# dataframe of the form ["citing_paper", "cited_paper", "citing_paper_year", "cited_paper_year"]

# We perform an aggregation of 1's (i.e. 1 citation) for all records where citing_paper_year - cited_paper_year <= 3.

#---------- Imports ------------- #
import sys
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
# Functions to effectively handle data 
# manipulation for DataFrames
import pyspark.sql.functions as F
# Diagnostics
import time
# -------------------------------- #
# Print output message if not run correctly
###########################################
if len(sys.argv) < 2:
	print ('\nUsage: CC.py <input_file> <(optional)num_partitions> <(optional) limit_year>\n')
	sys.exit(0)
###########################################

############################################
# Read input parameters
input_file  = sys.argv[1]
# subject_file = sys.argv[2]

# Set default partitions to 1
try:
	num_partitions = int(sys.argv[2])
except:
	num_partitions = 1
try:
	 limit_year = int(sys.argv[3])
except:
	 limit_year = None
# Set the mode by default as local. 
# If data is read from hdfs we switch to cluster
mode = 'local'
if input_file.startswith('hdfs://'):
	mode = 'distributed'
############################################
#####################################################################################################
program_name = "Spark CC" if not  limit_year else ("Spark " + str( limit_year) + "-year CC")
# Initialize spark context and spark-specific params
if  mode == 'local':
	conf = SparkConf().setMaster('local').setAppName(program_name)
else:
	conf = SparkConf().setAppName(program_name)
	
conf.set("spark.sql.crossJoin.enabled", True)
# Pass configuration to spark context
sc = SparkContext(conf = conf)
# Session is needed for spark.sql library, in order to use DataFrames
if mode == 'local':
	sp_session = SparkSession.builder.master('local').appName(program_name).getOrCreate()
else:
	sp_session = SparkSession.builder.appName(program_name).getOrCreate()
# Turn off logging (possible values: OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL
sc.setLogLevel('OFF')
#####################################################################################################
# Read the initial input
if mode == 'local':
	input_data   = sc.textFile('file:' + input_file)
	
elif mode == 'distributed':
	print("\n\nReading input from hdfs\n\n")
	input_data   = sc.textFile(input_file)
#####################################################################################################	
print("Mode is: " + mode)
initialisation_time = time.time()
print ("Num Partitions: " + str(num_partitions))
print ("Limit year: " + str(limit_year))
print ("\n\n")
# Initialise SPARK Data
print("Getting outlinks...\n")
# Read input data. Column "cited_papers" will be an array based on the splitting performed
outlinks = input_data.map(lambda x: (x.split("\t")[0].strip(), x.split("\t")[1].rsplit("|",2)[0].split(","), x.split("\t")[-1].split("|")[0].split("-")[0])).toDF(['paper', 'cited_papers', 'citing_paper_year']).cache()

print("Keeping only papers that cite others...\n")
# Create a DataFrame with nodes filtered based on whether they cite others or not. We are keeping only those that cite other papers
outlinks_actual = outlinks.filter(outlinks['cited_papers'][0] != "0").select('paper', F.explode(F.col('cited_papers')).alias('cited_paper'), F.col('citing_paper_year').cast('int')).repartition(num_partitions, 'paper').cache()

# If offset year is given, we need to perform some filtering of citations based on pub year. Proceed by normally calculating the 3-year based CC
if  limit_year:
	# We now need to filter out those records where citing year - cited year >  limit_year
	# a. join again with years, based on cited paper year - create a clone of the initial dataframe, because otherwise there will be an error due to similar column names
	cited_paper_years = outlinks.select('paper', 'citing_paper_year').withColumnRenamed('paper', 'cited_paper').withColumnRenamed('citing_paper_year', 'cited_paper_year')
	valid_citations   = outlinks_actual.join(cited_paper_years, outlinks_actual.cited_paper == cited_paper_years.cited_paper)\
					 .select(outlinks_actual.paper, cited_paper_years.cited_paper, outlinks_actual.citing_paper_year, cited_paper_years.cited_paper_year).repartition(num_partitions, 'paper')					

	# b. Filter out those where citing paper year > cited paper year + 3
	valid_citations = valid_citations.filter(valid_citations['citing_paper_year']-valid_citations['cited_paper_year'] <=  limit_year).repartition(num_partitions, 'paper')
	# print ("Showing valid citations")
	# valid_citations.show(100, False)

else:
	valid_citations = outlinks_actual

# Group by cited_paper and get counts
print("Counting citations...\n")
valid_citations = valid_citations.repartition(num_partitions, 'cited_paper').groupBy('cited_paper').count()
# valid_citations.show(20, False)
# Add papers which aren't cited
print("Joining with initial data...\n")
# Join with papers that aren't cited
valid_citations = valid_citations.join(outlinks, outlinks.paper == valid_citations.cited_paper, 'right_outer')\
				.select('paper', 'count')\
				.fillna(0).repartition(num_partitions, 'paper').cache()

# print("Number of records:\n")
# print(str(valid_citations.count()))
# print("Showing top cited:\n")
# valid_citations.sort(F.col('count').desc()).show(100, False)

########################################################################################################################################################
## WRITE THE OUTPUT OF CC
prefix = "CC_"
if limit_year:
	prefix = str(limit_year) + "-year_" + prefix
# Write output to file
if mode == 'local':
	input_file_name = input_file.split("/")[-1]
	valid_citations.write.format('csv').mode("overwrite").option("delimiter","\t").save(prefix + input_file_name)
	
elif mode == 'distributed':
	input_file_name = input_file.split("/")[-1]
	output_prefix = "/".join(input_file.split("/")[:-1])
	print ("Writing output at: ")
	print (output_prefix + "/" + prefix + input_file_name)
	valid_citations.write.mode("overwrite").option("delimiter","\t").csv(output_prefix + "/" + prefix + input_file_name)
########################################################################################################################################################

print ("\n\nDone!\n\n")
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################

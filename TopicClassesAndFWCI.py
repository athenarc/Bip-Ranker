"""
BIP Topics - Impact Class and FWCI Calculation

This script:
1. Loads publication scores (pagerank, attrank, citations) and concept mappings
2. Computes Field-Weighted Citation Impact (FWCI) metrics
3. Calculates topic-based impact class thresholds per concept
4. Assigns 5-point impact classes (C1-C5) for each metric
5. Writes output files: topic-based classes and FWCI metrics
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import logging
import argparse

# ============================================================================
# SETUP
# ============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate BIP topic-based impact classes and FWCI metrics')
parser.add_argument('--scores-file', 
                    default='/tmp/schatz/bip_metadata/output/doi_to_scores.csv',
                    help='Input file with publication scores (openaire_id, pid, type, year, pagerank, attrank, cc, 3y-cc)')
parser.add_argument('--concepts-file',
                    default='/tmp/schatz/bip_metadata/doi_to_concept_id_score.csv',
                    help='Input file with DOI to concept mappings')
parser.add_argument('--openaire-concepts-output',
                    default='/tmp/schatz/bip_metadata/openaire_id_to_concept_id_score.csv',
                    help='Output file for OpenAIRE ID to concept mappings')
parser.add_argument('--output-dir',
                    default='/tmp/schatz/bip_metadata/output/',
                    help='Output directory for all generated files')

args = parser.parse_args()

# File paths from arguments
scores_file = args.scores_file
concepts_file = args.concepts_file
openaire_concepts_output_file = args.openaire_concepts_output
output_dir = args.output_dir if args.output_dir.endswith('/') else args.output_dir + '/'

# Initialize Spark session
spark = SparkSession.builder.appName('BIP-topics').getOrCreate()
log4j = spark._jvm.org.apache.log4j
log4j.LogManager.getRootLogger().setLevel(log4j.Level.WARN)

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

# Read scores (openaire_id, pid, type, year, pagerank, attrank, cc, 3y-cc)
scores_raw = spark.read.options(header='True', inferSchema='True', delimiter='\t').csv(scores_file)
scores_raw = scores_raw.select(
    F.col('openaire_id').cast(StringType()),
    F.col('pid').cast(StringType()),
    F.col('type').cast(StringType()),
    F.col('year').cast(StringType()),
    F.col('pagerank').cast(DoubleType()),
    F.col('attrank').cast(DoubleType()),
    F.col('cc').cast(DoubleType()),
    F.col('3y-cc').cast(DoubleType())
)

# Mapping between openaire_id and DOI (pid)
openaire_to_pid = scores_raw.select('openaire_id', 'pid').cache()

# Aggregate metrics to openaire_id level (assumption: all records for a given openaire_id have the same metrics)
scores = scores_raw.groupBy('openaire_id').agg(
    F.max('pagerank').alias('pagerank'),
    F.max('attrank').alias('attrank'),
    F.max('cc').alias('cc'),
    F.max('3y-cc').alias('3y-cc'),
    F.first('type').alias('type'),
    F.first('year').alias('year')
)

# Read DOI -> concept mapping and map to openaire_id, keeping max confidence per concept
concepts_doi = spark.read.options(header='False', delimiter='\t').csv(concepts_file)
concepts_doi = concepts_doi.toDF('doi', 'concept', 'confidence')
concepts_doi = concepts_doi.withColumn('confidence', F.col('confidence').cast(DoubleType()))

# Join concepts on DOI (pid) to get openaire_id
concepts_with_open = concepts_doi.join(openaire_to_pid, concepts_doi.doi == openaire_to_pid.pid, 'inner')

# Keep unique concepts with max confidence per openaire_id (input file already filtered to >= 0.3)
concepts = concepts_with_open.groupBy('openaire_id', 'concept').agg(F.max('confidence').alias('confidence'))

# Keep up to top 3 concepts per openaire_id by confidence
top3_window = Window.partitionBy('openaire_id').orderBy(F.col('confidence').desc())
concepts = concepts.withColumn('rn', F.row_number().over(top3_window)).filter(F.col('rn') <= 3).drop('rn')

# Persist an output file for openaire_id -> concept (max confidence)
concepts.select('openaire_id', 'concept', 'confidence')\
    .orderBy('openaire_id', 'confidence')\
    .write.mode('overwrite').options(header='False', delimiter='\t').csv(openaire_concepts_output_file)

# Build working dataframe at openaire_id level and keep same downstream column name 'id'
d = concepts.join(scores, 'openaire_id').repartition(64, "openaire_id").select(
    F.col('openaire_id').alias('id'), 'concept', 'pagerank', 'attrank', 'cc', '3y-cc', 'type', 'year'
).cache()

# ============================================================================
# COMPUTE FIELD-WEIGHTED CITATION IMPACT (FWCI)
# ============================================================================

# Compute Field-Weighted Citation Impact (FWCI) for total citations
expected_citations = d.filter(F.col('cc').isNotNull()) \
    .groupBy('concept', 'type', 'year') \
    .agg(F.avg('cc').alias('expected_citations')) \
    .filter(F.col('expected_citations') > 0)  # Avoid division by zero

# Compute Field-Weighted Citation Impact for 3-year citations (3y-FWCI)
expected_citations_3y = d.filter(F.col('3y-cc').isNotNull()) \
    .groupBy('concept', 'type', 'year') \
    .agg(F.avg('3y-cc').alias('expected_citations_3y')) \
    .filter(F.col('expected_citations_3y') > 0)  # Avoid division by zero

# Join with original data and compute both FWCI metrics
d_with_fwci = d.join(expected_citations, ['concept', 'type', 'year'], 'left') \
    .join(expected_citations_3y, ['concept', 'type', 'year'], 'left') \
    .withColumn('fwci', 
               F.when(F.col('expected_citations').isNull() | (F.col('expected_citations') == 0), 
                      F.lit(None))
               .otherwise(F.col('cc') / F.col('expected_citations'))) \
    .withColumn('3y-fwci', 
               F.when(F.col('expected_citations_3y').isNull() | (F.col('expected_citations_3y') == 0), 
                      F.lit(None))
               .otherwise(F.col('3y-cc') / F.col('expected_citations_3y')))

# Replace d with the enhanced version
d.unpersist()
d = d_with_fwci.cache()

# ============================================================================
# CALCULATE IMPACT CLASS THRESHOLDS PER CONCEPT
# ============================================================================

print("concept_id\tpagerank_top001\tpagerank_top01\tpagerank_top1\tpagerank_top10\tattrank_top001\tattrank_top01\tattrank_top1\tattrank_top10\t3-cc_top001\t3-cc_top01\t3-cc_top1\t3-cc_top10\tcc_top001\tcc_top01\tcc_top1\tcc_top10")

# Define metrics to calculate thresholds and classes for
metrics = ["pagerank", "attrank", "3y-cc", "cc"]

# Get the count of papers per concept
concept_counts = d.groupBy('concept').agg(F.count('*').alias('num_nodes'))

# Calculate offset positions (minimum 1)
concept_counts = concept_counts.withColumn('top_001_offset', 
    F.when(F.floor(F.col('num_nodes') * 0.0001) == 0, 1).otherwise(F.floor(F.col('num_nodes') * 0.0001)))
concept_counts = concept_counts.withColumn('top_01_offset', 
    F.when(F.floor(F.col('num_nodes') * 0.001) == 0, 1).otherwise(F.floor(F.col('num_nodes') * 0.001)))
concept_counts = concept_counts.withColumn('top_1_offset', 
    F.when(F.floor(F.col('num_nodes') * 0.01) == 0, 1).otherwise(F.floor(F.col('num_nodes') * 0.01)))
concept_counts = concept_counts.withColumn('top_10_offset', 
    F.when(F.floor(F.col('num_nodes') * 0.1) == 0, 1).otherwise(F.floor(F.col('num_nodes') * 0.1)))

# Join offsets to main dataframe
d = d.join(concept_counts, 'concept', 'left')

# For each metric, find thresholds using distinct scores with cumulative counts (like old approach)
thresholds_list = []
for metric in metrics:
    # Get distinct scores with counts per concept (like old approach)
    distinct_scores = d.groupBy('concept', metric).agg(F.count('*').alias('count'))
    
    # Add cumulative count per concept (ordered by metric descending)
    window_spec = Window.partitionBy('concept').orderBy(F.col(metric).desc()).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    distinct_scores = distinct_scores.withColumn('cumulative', F.sum('count').over(window_spec))
    
    # Join with offsets
    distinct_scores = distinct_scores.join(concept_counts.select('concept', 'top_001_offset', 'top_01_offset', 'top_1_offset', 'top_10_offset'), 'concept', 'left')
    
    # Find minimum score where cumulative <= offset for each threshold
    thresholds = distinct_scores.groupBy('concept').agg(
        F.min(F.when(F.col('cumulative') <= F.col('top_001_offset'), F.col(metric))).alias('{}_top001'.format(metric)),
        F.min(F.when(F.col('cumulative') <= F.col('top_01_offset'), F.col(metric))).alias('{}_top01'.format(metric)),
        F.min(F.when(F.col('cumulative') <= F.col('top_1_offset'), F.col(metric))).alias('{}_top1'.format(metric)),
        F.min(F.when(F.col('cumulative') <= F.col('top_10_offset'), F.col(metric))).alias('{}_top10'.format(metric))
    )
    
    thresholds_list.append(thresholds)

# Merge all threshold dataframes
thresholds_df = thresholds_list[0]
for thresholds in thresholds_list[1:]:
    thresholds_df = thresholds_df.join(thresholds, 'concept', 'outer')

# Join all thresholds back to main dataframe
d = d.join(thresholds_df, 'concept', 'left')

# ============================================================================
# ASSIGN IMPACT CLASSES BASED ON THRESHOLDS
# ============================================================================

# Assign classes for all metrics
for metric in metrics:
    d = d.withColumn('{}_five_point_class'.format(metric), F.lit('C5'))
    d = d.withColumn('{}_five_point_class'.format(metric), 
                     F.when(F.col(metric) >= F.col("{}_top10".format(metric)), F.lit('C4')).otherwise(F.col('{}_five_point_class'.format(metric))))
    d = d.withColumn('{}_five_point_class'.format(metric), 
                     F.when(F.col(metric) >= F.col("{}_top1".format(metric)), F.lit('C3')).otherwise(F.col('{}_five_point_class'.format(metric))))
    d = d.withColumn('{}_five_point_class'.format(metric), 
                     F.when(F.col(metric) >= F.col("{}_top01".format(metric)), F.lit('C2')).otherwise(F.col('{}_five_point_class'.format(metric))))
    d = d.withColumn('{}_five_point_class'.format(metric), 
                     F.when(F.col(metric) >= F.col("{}_top001".format(metric)), F.lit('C1')).otherwise(F.col('{}_five_point_class'.format(metric))))

# Print limits for all concepts
limits_df = d.select('concept', 
                     'pagerank_top001', 'pagerank_top01', 'pagerank_top1', 'pagerank_top10',
                     'attrank_top001', 'attrank_top01', 'attrank_top1', 'attrank_top10',
                     '3y-cc_top001', '3y-cc_top01', '3y-cc_top1', '3y-cc_top10',
                     'cc_top001', 'cc_top01', 'cc_top1', 'cc_top10').distinct().orderBy('concept')

for row in limits_df.collect():
    print('\t'.join(map(str, row)))

# ============================================================================
# WRITE OUTPUT FILES
# ============================================================================

# Write FWCI for OpenAIRE IDs
logger.info("Writing FWCI for OpenAIRE IDs")
d.select(
    F.col("id").alias("openaire_id"),
    F.col("concept").alias("concept"),
    F.col("fwci").alias("fwci")
).write.options(header='False', delimiter='\t', compression='gzip', nullValue='').mode('overwrite').csv(output_dir + "/bip-db/" + "FWCI_openaire_ids.txt.gz")

# Write 3y-FWCI for OpenAIRE IDs
logger.info("Writing 3y-FWCI for OpenAIRE IDs")
d.select(
    F.col("id").alias("openaire_id"),
    F.col("concept").alias("concept"),
    F.col("3y-fwci").alias("3y-fwci")
).write.options(header='False', delimiter='\t', compression='gzip', nullValue='').mode('overwrite').csv(output_dir + "/bip-db/" + "3-year_FWCI_openaire_ids.txt.gz")

# Prepare data with PIDs for DOI-based outputs (inner join to keep only papers with PIDs)
d_with_pid = d.join(openaire_to_pid, d.id == openaire_to_pid.openaire_id, 'inner') \
    .drop('id') \
    .withColumnRenamed('pid', 'id')

logger.info("Total rows with valid PIDs: %d", d_with_pid.count())

# Write topic-based classes output
logger.info("Writing topic-based classes output")
d_with_pid.select(
    F.col("id").alias("identifier"),
    F.col("concept").alias("concept"),
    F.col("pagerank_five_point_class").alias("pagerank_class"),
    F.col("attrank_five_point_class").alias("attrank_class"),
    F.col("3y-cc_five_point_class").alias("3y-cc_class"),
    F.col("cc_five_point_class").alias("cc_class")
).write.options(header='True', delimiter='\t').mode('overwrite').csv(output_dir + "/topics/")

# Write FWCI for PIDs (DOIs)
logger.info("Writing FWCI for PIDs")
d_with_pid.select(
    F.col("id").alias("identifier"),
    F.col("concept").alias("concept"),
    F.col("fwci").alias("fwci")
).write.options(header='False', delimiter='\t', compression='gzip', nullValue='').mode('overwrite').csv(output_dir + "/bip-db/" + "FWCI.txt.gz")

# Write 3y-FWCI for PIDs (DOIs)
logger.info("Writing 3y-FWCI for PIDs")
d_with_pid.select(
    F.col("id").alias("identifier"),
    F.col("concept").alias("concept"),
    F.col("3y-fwci").alias("3y-fwci")
).write.options(header='False', delimiter='\t', compression='gzip', nullValue='').mode('overwrite').csv(output_dir + "/bip-db/" + "3-year_FWCI.txt.gz")

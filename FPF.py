from pyspark.mllib.fpm import FPGrowth
from pyspark import SparkContext, SparkConf
import csv

conf = SparkConf().setAppName("FPF")
sc = SparkContext(conf=conf)
#sc = SparkContext
data = sc.textFile("/Users/yinli/software/spark-2.0.1-bin-hadoop2.7/data/mllib/sample_fpgrowth_copy.txt")
#data = sc.textFile("/Users/yinli/Downloads/fpgrowth-master/5000-out1.csv");
#data = sc.textFile("/Users/yinli/Downloads/fpgrowth-master/5000-out1.csv").map(lambda line: line.split(",")).filter(lambda line: len(line)>1).map(lambda line: (line[0],line[1])).collect()
transactions = data.map(lambda line: line.strip().split(','))
print transactions
for fi in transactions.collect():
    del transactions.collect()[0]
    #print(fi)
model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=1)
result = model.freqItemsets().collect()
for fi in result:
    print(fi)
from pyspark.mllib.fpm import FPGrowth
from pyspark import SparkContext, SparkConf
import csv

conf = SparkConf().setAppName("FPF")
#conf = conf.setMaster("local[*]")
sc = SparkContext(conf=conf)

sc.setLogLevel("ERROR")
#sc = SparkContext
data = sc.textFile("/Users/yinli/Downloads/fpgrowth-master/5000-out1.txt")
#data = sc.textFile("/Users/yinli/Downloads/fpgrowth-master/5000-out1.csv");
#data = sc.textFile("/Users/yinli/Downloads/fpgrowth-master/5000-out1.csv").map(lambda line: line.split(",")).filter(lambda line: len(line)>1).map(lambda line: (line[0],line[1])).collect()
transactions = data.map(lambda line: line.strip().split('\t'))
#print transactions
#for fi in transactions.collect():
 #   del fi[0]
  #  print(fi)
model = FPGrowth.train(transactions, minSupport=0.01, numPartitions=2)
result = model.freqItemsets().collect()
for fi in result:
    print(fi)
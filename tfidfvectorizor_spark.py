from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram, \
    IDFModel, CountVectorizer, Normalizer
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


sc = SparkContext('local')
spark = SparkSession(sc)

'''
This is an example of how I implemented Sklearn's TFIDFVectorizor
fit transform class into PySpark. The class is simply a function for 
brevity.

The example given at
https://spark.apache.org/docs/latest/ml-features.html#tf-idf
demonstrates TFIDF using the hasing trick.

This technique although memory efficient is not reversable.
This means we will not be able to see the relationship between words 
and tfidf scores.

In addition the example is missing the post normalization step
explained by Chris Manning here

https://www.youtube.com/watch?v=ZEkO8QSlynY @ 7:04

Credit again to Chris Manning for explaining the TFIDF concept.

'''


def create_tfidf_model(sentenceDataFrame, ngrams=1, minDocFreq=0):

    tokenized = Tokenizer(
        inputCol="text",
        outputCol="words").transoform(sentenceDataFrame)

    ngramDataFrame = NGram(
        n=ngrams,
        inputCol="words",
        outputCol="ngrams").transform(tokenized)

    countVect = CountVectorizer(
        inputCol="ngrams",
        outputCol="rawFeatures")

    countVectModel = countVect.fit(ngramDataFrame)

    featurizedData = countVectModel.transform(ngramDataFrame)

    idf = IDF(
        minDocFreq=minDocFreq,
        inputCol="rawFeatures",
        outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("label", "features")

    normalizer = Normalizer(inputCol="features", outputCol='scores')
    X = normalizer.transform(rescaledData)

    return X

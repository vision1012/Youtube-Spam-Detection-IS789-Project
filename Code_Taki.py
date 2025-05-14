
# Importing necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, month, to_date
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
spark = SparkSession.builder.appName("YouTubeSpamDetection-Master").getOrCreate()


#Load Dataset Clean data
files = ["Youtube01-Psy.csv", "Youtube02-KatyPerry.csv", "Youtube03-LMFAO.csv", "Youtube04-Eminem.csv", "Youtube05-Shakira.csv"]

df = spark.read.csv(files[0], header=True, inferSchema=True)
for file in files[1:]:
    df_next = spark.read.csv(file, header=True, inferSchema=True)
    df = df.union(df_next)

df = df.withColumn("CLASS", col("CLASS").cast(IntegerType()))
df_clean = df.na.drop(subset=["CLASS"])
df_clean.select("CLASS").distinct().show()

df_preprocessed = df_clean.select("CONTENT", "CLASS")
df_preprocessed = df_preprocessed.withColumn("CONTENT", lower(col("CONTENT")))
df_preprocessed = df_preprocessed.withColumn("CONTENT", regexp_replace(col("CONTENT"), r"http\S+|www\S+|https\S+", "URL"))
df_preprocessed = df_preprocessed.withColumn("CONTENT", regexp_replace(col("CONTENT"), r"[^a-zA-Z\s]", ""))
df_preprocessed = df_preprocessed.withColumn("CONTENT", regexp_replace(col("CONTENT"), r"\s+", " "))

tokenizer = Tokenizer(inputCol="CONTENT", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="features", numFeatures=1000)

pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf])
pipeline_model = pipeline.fit(df_preprocessed)
df_features = pipeline_model.transform(df_preprocessed)


total_count = df_features.count()
class_counts = df_features.groupBy("CLASS").count().collect()

fractions = {}
for row in class_counts:
    fractions[row["CLASS"]] = 0.8

train_data = df_features.sampleBy("CLASS", fractions=fractions, seed=42)
test_data = df_features.subtract(train_data)



# Train the model NaiveBayes
nb = NaiveBayes(featuresCol="features", labelCol="CLASS")
nb_model = nb.fit(train_data)
nb_predictions = nb_model.transform(test_data)

# Evaluate metrics
evaluator = MulticlassClassificationEvaluator(labelCol="CLASS", predictionCol="prediction")

nb_accuracy = evaluator.evaluate(nb_predictions, {evaluator.metricName: "accuracy"})
nb_precision = evaluator.evaluate(nb_predictions, {evaluator.metricName: "weightedPrecision"})
nb_recall = evaluator.evaluate(nb_predictions, {evaluator.metricName: "weightedRecall"})
nb_f1 = evaluator.evaluate(nb_predictions, {evaluator.metricName: "f1"})

print("Naive Bayes Metrics:")
print(f"Accuracy: {nb_accuracy}")
print(f"Precision: {nb_precision}")
print(f"Recall: {nb_recall}")
print(f"F1 Score: {nb_f1}")


# ### Logistic Regression
#Train Model
lr = LogisticRegression(featuresCol="features", labelCol="CLASS")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

# Evaluate metrics
evaluator = MulticlassClassificationEvaluator(labelCol="CLASS", predictionCol="prediction")

lr_accuracy = evaluator.evaluate(lr_predictions, {evaluator.metricName: "accuracy"})
lr_precision = evaluator.evaluate(lr_predictions, {evaluator.metricName: "weightedPrecision"})
lr_recall = evaluator.evaluate(lr_predictions, {evaluator.metricName: "weightedRecall"})
lr_f1 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "f1"})

print("Logistic Regression Metrics:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1 Score: {lr_f1}")


# ### Decision Tree
#Train Model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="CLASS")
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

# Evaluate metrics
evaluator = MulticlassClassificationEvaluator(labelCol="CLASS", predictionCol="prediction")

dt_accuracy = evaluator.evaluate(dt_predictions, {evaluator.metricName: "accuracy"})
dt_precision = evaluator.evaluate(dt_predictions, {evaluator.metricName: "weightedPrecision"})
dt_recall = evaluator.evaluate(dt_predictions, {evaluator.metricName: "weightedRecall"})
dt_f1 = evaluator.evaluate(dt_predictions, {evaluator.metricName: "f1"})

print("Decision Tree Metrics:")
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1 Score: {dt_f1}")


# ### Random Forest

#Train Model
rf = RandomForestClassifier(featuresCol="features", labelCol="CLASS", numTrees=20)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

# Evaluate metrics
evaluator = MulticlassClassificationEvaluator(labelCol="CLASS", predictionCol="prediction")

rf_accuracy = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})
rf_precision = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedPrecision"})
rf_recall = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedRecall"})
rf_f1 = evaluator.evaluate(rf_predictions, {evaluator.metricName: "f1"})

print("Random Forest Metrics:")
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1 Score: {rf_f1}")



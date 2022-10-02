
# Titanic: Machine Learining from Disaster (Part III)

### Abstract

**The aim of the third part of the project**

This is the third and also the final part of the Titanic project, in which we build a machine learning model to predict the survivability of passengers of the Titanic shipwreck, based on the data that we've already processed in the second part. 

### Introduction

**Overview**

![image1](https://www.ionos.com/digitalguide/fileadmin/DigitalGuide/Teaser/machine-learning-t.jpg)

After Data Analysis step in the first part and Data Preprocessing step in second part, in the third part of the Titanic project, we are going to train a machine learning model to predict whether a passenger survived the notorious maritime disaster or not. We will evaluate our model performance based on various measures to select the most suitable one that gives us the best result.

### Data Source

**Data used in the part III of the project**

We are going use the cleaned Titanic data set that we prepared in the second part as our data for both model training and testing. For final predictions, we will use the ```test.csv``` data set that we reserved from the beginning. 

The path to both data set can be found in the project repository. 

### Break down of the third part of the project

**Part III of our project is structured as follows:**

1. Installing, importing libraries and data used in our project;

2. Performing model selection, model training, testing and evaluating performance;

3. Using the model to make predictions on the test data;

4. The last section is the final recap and conclusion of the project.


### 1. Import libraries, data set

#### 1.1 Import the packages

We first start by loading all the libraries used in the third part of the project:

```Julia
# Load the packages to the current working environment
using CSV # To handle CSV file
using PrettyPrinting # For pretty printing
using DataFrames # Data frame in Julia
using MLJ # ML interface in Julia
using CairoMakie # For plotting
using StableRNGs # Stable seeds
```

#### 1.2. Load the data

For the data, we can load the ```.csv``` file to the environment and convert to standard ```DataFrame``` format. As mentioned before, we use the prepared train data set:

```Julia
# Provide path to our data. Our data set can be found in project's repository
path = "titanic_clean.csv";

# Load our data, convert from .csv format to DataFrame
titanic = DataFrame(CSV.File(path));
```

### 2. Machine learning model building

#### 2.1. Scientific types coercion

If you recalled, in the second part, in order to use the missing value imputer, we have already converted the machine types of the variables in the data to scientific types. Unfortunately, these properties did not preserved after saving and loading the data set again, so we have to coerce the data one more time. We first inspect metadata information, including column machine types and then coerce into proper scientific types:

```Julia
# Inspect current machine types of our data
schema(titanic)
```

```Julia
┌────────────┬────────────┬─────────┐
│ names      │ scitypes   │ types   │
├────────────┼────────────┼─────────┤
│ Survived   │ Count      │ Int64   │
│ Pclass     │ Count      │ Int64   │
│ Sex        │ Textual    │ String7 │
│ Age        │ Continuous │ Float64 │
│ Fare       │ Continuous │ Float64 │
│ Embarked   │ Textual    │ String1 │
│ FamSize    │ Count      │ Int64   │
│ Salutation │ Textual    │ String7 │
└────────────┴────────────┴─────────┘
```

So currently our data are either ```String``` (```Textual```) or ```Numeric``` (```Float64``` and ```Int64```). Let's convert them into proper scientific types and then double-check to make sure everything is correct:

```Julia
# Coerce data into proper scientific types
titanic = coerce(titanic, :Survived => OrderedFactor, 
                                 :Pclass => OrderedFactor,
                                 :Sex => Multiclass,
                                 :Age => Continuous,
                                 :Fare => Continuous,
                                 :Embarked => Multiclass,
                                 :FamSize => Count,
                                 :Salutation => Multiclass);

# Double-check types of our data after converting
schema(titanic)
```

```Julia
┌────────────┬──────────────────┬───────────────────────────────────┐
│ names      │ scitypes         │ types                             │
├────────────┼──────────────────┼───────────────────────────────────┤
│ Survived   │ OrderedFactor{2} │ CategoricalValue{Int64, UInt32}   │
│ Pclass     │ OrderedFactor{3} │ CategoricalValue{Int64, UInt32}   │
│ Sex        │ Multiclass{2}    │ CategoricalValue{String7, UInt32} │
│ Age        │ Continuous       │ Float64                           │
│ Fare       │ Continuous       │ Float64                           │
│ Embarked   │ Multiclass{3}    │ CategoricalValue{String1, UInt32} │
│ FamSize    │ Count            │ Int64                             │
│ Salutation │ Multiclass{7}    │ CategoricalValue{String7, UInt32} │
└────────────┴──────────────────┴───────────────────────────────────┘
```

Let's also quickly look for the levels of our ```OrderedFactor``` variables:

```Julia
# For Survived column (target variable)
levels(titanic.Survived)
```

```Julia
2-element Vector{Int64}:
 0
 1
```

Since we consider survived as positive output and not-survived as negative one, we therefore swap the level of our factor variables and re-inspect after doing so:

```Julia
# refactor the level of our categorical variable
levels!(titanic.Survived, [1, 0])

# Reinspect the level
levels(titanic.Survived);
```

```Julia
2-element Vector{Int64}:
 1
 0
```
We can do the same thing for ```Pclass``` variable:

```Julia
# For Pclass variable (feature variable)
levels(titanic.Pclass)
```

```Julia
3-element Vector{Int64}:
 1
 2
 3
```

The level is in the correct order this time, so we don't have to do anything. 

#### 2.2 Data transformation

To feed our data to a model, we have to split them into ```target``` and ```feature``` variables. We set a seed to make it reproducible and also shuffle data along the split:

```Julia
# Initialize a stable seed
rng = StableRNG(365);

# Split the data into target and features, shuffle along the way
target, features = unpack(titanic, ==(:Survived), shuffle = true, rng = rng)
```

```Julia
(CategoricalArrays.CategoricalValue{Int64, UInt32}[0, 0, 0, 1, 0, 0, 1, 0, 0, 1  …  0, 0, 1, 0, 1, 0, 0, 0, 1, 1], 891×7 DataFrame
 Row │ Pclass  Sex     Age      Fare     Embarked  FamSize  Salutation 
     │ Cat…    Cat…    Float64  Float64  Cat…      Int64    Cat…
─────┼─────────────────────────────────────────────────────────────────
   1 │ 3       male       26.0  20.575   S               3  Mr
  ⋮  │   ⋮       ⋮        ⋮        ⋮        ⋮         ⋮         ⋮
                                                       890 rows omitted)
```

Since a lot of algorithms cannot handle categorical data, we have to find a way to encode them. Also, since the continuous variables are all measured in different scales (like ```FamSize``` and ```Fare```), standardization the data is needed. So for the features data, we define a transformation pipeline as follows.

* Standardize continuous variables;

* Continuous encoding categorical variables.

More about transformers implemented in MLJ can be found in the documentation [here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/).

```Julia
# Load the standardizer and encoder
@load Standardizer;
@load ContinuousEncoder;

# Build a transformation pipeline
SimplePipe = Pipeline(
    Standardizer(),
    ContinuousEncoder()
)
```

```Julia
UnsupervisedPipeline(
  standardizer = Standardizer(
        features = Symbol[],
        ignore = false,
        ordered_factor = false,
        count = false),
  continuous_encoder = ContinuousEncoder(
        drop_last = false,
        one_hot_ordered_factors = false),
  cache = true)
```

Using the machine, we connect the pipeline to the feature data and then  apply the transformations:

```Julia
# Connect our pipeline to the data
trans = machine(SimplePipe, features)
```

```Julia
untrained Machine; caches model-specific representations of data
  model: UnsupervisedPipeline(standardizer = Standardizer(features = Symbol[], …), …)
  args:
    1:  Source @132 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Count}, AbstractVector{Multiclass{2}}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{7}}, AbstractVector{OrderedFactor{3}}}}
```

```Julia
# Train the pipeline using the feature data
MLJ.fit!(trans)
```

```Julia
[ Info: Training machine(UnsupervisedPipeline(standardizer = Standardizer(features = Symbol[], …), …), …).
[ Info: Training machine(Standardizer(features = Symbol[], …), …).
[ Info: Training machine(ContinuousEncoder(drop_last = false, …), …).
trained Machine; caches model-specific representations of data
  model: UnsupervisedPipeline(standardizer = Standardizer(features = Symbol[], …), …)
  args:
    1:  Source @132 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Count}, AbstractVector{Multiclass{2}}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{7}}, AbstractVector{OrderedFactor{3}}}}
```

```Julia
# Transform, re-assign the feature data
features = MLJ.transform(trans, features);

# Inspect our transformed data
first(features, 5) 
```

```Julia
5×16 DataFrame
 Row │ Pclass   Sex__female  Sex__male  Age        Fare        Embarked__C  Embarked__Q  Embarked__S  FamSize  Salutation__Dr  Salutation__Master  Salutation__Miss  Salutation__Mr ⋯
     │ Float64  Float64      Float64    Float64    Float64     Float64      Float64      Float64      Float64  Float64         Float64             Float64           Float64        ⋯
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     3.0          0.0        1.0  -0.258192  -0.234019           0.0          0.0          1.0      3.0             0.0                 0.0               0.0             1.0 ⋯
  ⋮  │    ⋮          ⋮           ⋮          ⋮          ⋮            ⋮            ⋮            ⋮          ⋮           ⋮                 ⋮                  ⋮                ⋮        ⋱
                                                                                                                                                         3 columns and 4 rows omitted
```

The data is ready to use for model selection and model training.

#### 2.3. Model selection

MLJ supports a wide range of machine learning algorithms included from third-party packages, since it has metada for model registries. To see all models which are compatible with our data, we can match the data with the model metadata, apply any conditions if necessary. For this project, we will use models that return a probabilistic prediction type and were written in pure Julia.

```Julia
# Model matching
models() do model
    matching(model, features, target) &&
    model.prediction_type == :probabilistic &&
    model.is_pure_julia
end
```

```Julia
20-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
 (name = AdaBoostStumpClassifier, package_name = DecisionTree, ... )
 (name = BayesianLDA, package_name = MultivariateStats, ... )
 (name = BayesianSubspaceLDA, package_name = MultivariateStats, ... )
 (name = ConstantClassifier, package_name = MLJModels, ... )
 (name = DecisionTreeClassifier, package_name = BetaML, ... )
 (name = DecisionTreeClassifier, package_name = DecisionTree, ... )
 (name = EvoTreeClassifier, package_name = EvoTrees, ... )
 (name = GaussianNBClassifier, package_name = NaiveBayes, ... )
 (name = KNNClassifier, package_name = NearestNeighborModels, ... )
 (name = KernelPerceptronClassifier, package_name = BetaML, ... )
 (name = LDA, package_name = MultivariateStats, ... )
 (name = LinearBinaryClassifier, package_name = GLM, ... )
 (name = LogisticClassifier, package_name = MLJLinearModels, ... )
 (name = MultinomialClassifier, package_name = MLJLinearModels, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = DecisionTree, ... )
 (name = SubspaceLDA, package_name = MultivariateStats, ... )
```

There are in total 20 models that match with our target and feature data, which is quite a lot! Also, one algorithm is provided by two different packages (```RandomForestClassifier```). This is where specific domain knowledge comes in handy: since some linear models depends on computing and analyzing **positive definite matrix**, which may not be the case for our data, we would have actively exclude them from our set of models in order to avoid any error during the benchmark process. There are 3 of them in our set of models, namely ```NaiveBayes```, ```GLM``` and ```MLJLinearModels```

```Julia
# Exclude from models from our 
packs = ["NaiveBayes", "GLM", "MLJLinearModels"];
```

After exclude 3 models, we still so many options (exactly 16) to consider. To narrow down our choices and select one that best fits with our data, we can analyze their performances using a common task: we train each of them with default hyperparameters on the training set, evaluate the performance on the test set and then compare the their measures. We will use various measures that are commonly used to evaluate performance of classification task: **Accuracy**, **Cross-entropy loss**, **F1-score**. And also the **ROC curve** for a binary classifier system such as this.

To reserve some data for testing, we split the data into training and testing data based on indexes. We will use the common 70/30 split, which means 70% of data for training and the rest 30% for testing:

```Julia
# Split the data into training and testing
train, test = partition(eachindex(target), 0.7, shuffle = true, rng = rng)
```

```Julia
([170, 776, 59, 261, 205, 508, 680, 540, 348, 431  …  235, 631, 272, 122, 320, 798, 335, 236, 388, 801], [851, 623, 624, 672, 587, 43, 862, 826, 70, 848  …  245, 646, 7, 770, 860, 498, 638, 351, 384, 68])
```

Next, we create multiple empty vectors to store information, as well as measures of our models and their ROC curve. These include model name, the value of Accuracy, Cross-entropy loss and F1-score, all measured on the testing data, and one to store the plotting line:

```Julia
# Initialize empty array to store measures for our analysis
model_names = Vector{String}();

loss_acc = [];

loss_ce = [];

loss_f1 = [];

plot_line = [];
```

We also have to set up the plotting environment for the ROC curves, as we will put them in one plot for easy comparasion. For each model in the set of models, its ROC curve will be calculated and plotted:

```Julia
# Set up plotting environment for ROC curves
fig = Figure(resolution = (1920, 1080));

update_theme!(with_theme = theme_light(), fontsize = 30);

axis = fig[1, 1] = Axis(fig, title = "ROC curves",
                    xlabel = "False Positive Rate",
                    xticks = (0:0.2:1),
                    xlabelfont = "TeX Gyre Heros Makie Bold",
                    ylabel = "True Positive Rate",
                    yticks = (0:0.2:1),
                    ylabelfont = "TeX Gyre Heros Makie Bold");
```

The preparation steps are done. We can finally training and comparing our models. Note that this process might take some time to complete since we are training and testing a range of models:

```Julia
# Training, fitting and evaluating models performance
for i in models(matching(features, target))

    # Apply filter to find matching models with our data
    if i.prediction_type == :probabilistic && i.is_pure_julia == true && i.package_name ∉ pack
        
        model_name = i.name
        package_name = i.package_name
        eval(:(clf = @load $model_name pkg = $package_name verbosity = true))

        # Connect the matching model with data. Train on the training data
        clf_machine = machine(clf(), features, target)
        MLJ.fit!(clf_machine, rows = train)
        
        # Making predictions on the testing data. Draw the ROC curve
        target_pre = MLJ.predict(clf_machine, rows = test)
        fprs, tprs, thresholds = roc(target_pre, target[test])
        line = lines!(axis, fprs, tprs, linewidth = 5)

        # Calcutale measures to evaluate model performance
        ce_loss = mean(cross_entropy(target_pre, target[test]))
        acc = MLJ.accuracy(StatsBase.mode.(target_pre), target[test])
        f1_score = MLJ.f1score(StatsBase.mode.(target_pre), target[test])

        # Fill the empty vector with model information
        push!(plot_line, line)
        push!(model_names, i.name)
        append!(loss_acc, acc)
        append!(loss_ce, ce_loss)
        append!(loss_f1, f1_score)
    end
end
```

```Julia
[ Info: For silent loading, specify `verbosity=0`. 
import MLJDecisionTreeInterface ✔
[ Info: Training machine(AdaBoostStumpClassifier(n_iter = 10, …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import MLJMultivariateStatsInterface ✔
[ Info: Training machine(BayesianLDA(method = gevd, …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import MLJMultivariateStatsInterface ✔
[ Info: Training machine(BayesianSubspaceLDA(normalize = false, …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import MLJModels ✔
[ Info: Training machine(ConstantClassifier(), …).
[ Info: For silent loading, specify `verbosity=0`.
import BetaML ✔
[ Info: Training machine(DecisionTreeClassifier(maxDepth = 0, …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import MLJDecisionTreeInterface ✔
[ Info: Training machine(DecisionTreeClassifier(max_depth = -1, …), …).
[ Info: For silent loading, specify `verbosity=0`.
import EvoTrees ✔
[ Info: Following 15 arguments were not provided and will be set to default: nbins, alpha, gamma, nrounds, metric, max_depth, T, loss, lambda, min_weight, colsample, eta, rng, device, rowsample.
[ Info: Training machine(EvoTreeClassifier(loss = EvoTrees.Softmax(), …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import NearestNeighborModels ✔
[ Info: Training machine(KNNClassifier(K = 5, …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
[ Info: Training machine(KernelPerceptronClassifier(K = radialKernel, …), …).
Training Kernel Perceptron... 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:01:02        
[ Info: For silent loading, specify `verbosity=0`. 
import MLJMultivariateStatsInterface ✔
[ Info: Training machine(LDA(method = gevd, …), …).
[ Info: For silent loading, specify `verbosity=0`.
import MLJFlux ✔
[ Info: Training machine(NeuralNetworkClassifier(builder = Short(n_hidden = 0, …), …), …).
Optimising neural net: 100%[=========================] Time: 0:00:01
[ Info: For silent loading, specify `verbosity=0`.
import BetaML ✔
[ Info: Training machine(PegasosClassifier(initialθ = Float64[], …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
[ Info: Training machine(PerceptronClassifier(initialθ = Float64[], …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
[ Info: Training machine(RandomForestClassifier(nTrees = 30, …), …).
[ Info: For silent loading, specify `verbosity=0`. 
import MLJDecisionTreeInterface ✔
[ Info: Training machine(RandomForestClassifier(max_depth = -1, …), …).
[ Info: For silent loading, specify `verbosity=0`.
import MLJMultivariateStatsInterface ✔
[ Info: Training machine(SubspaceLDA(normalize = true, …), …).
```

Afterward, we can compare the performances of our models as well as inspect their ROC curves:

```Julia
# Inspect models performance. 
model_info = DataFrame(ModelName = model_names, Accuracy = loss_acc, CrossEntropyLoss = loss_ce, F1Score = loss_f1);

# Sorted by Cross-entropy loss
pprint(sort!(model_info,[:CrossEntropyLoss]))
```

```Julia
16×4 DataFrame
 Row │ ModelName                   Accuracy  CrossEntropyLoss  F1Score
     │ String                      Any       Any               Any
─────┼──────────────────────────────────────────────────────────────────
   1 │ BayesianSubspaceLDA         0.838951  0.41959           0.865204
   2 │ BayesianLDA                 0.838951  0.419872          0.865204
   3 │ EvoTreeClassifier           0.805243  0.426385          0.843373
   4 │ NeuralNetworkClassifier     0.771536  0.482559          0.807571
   5 │ AdaBoostStumpClassifier     0.816479  0.486544          0.846395
   6 │ KernelPerceptronClassifier  0.7603    0.552962          0.796178
   7 │ PegasosClassifier           0.700375  0.60906           0.791667
   8 │ SubspaceLDA                 0.82397   0.615012          0.847896
   9 │ ConstantClassifier          0.595506  0.67663           0.746479
  10 │ LDA                         0.82397   0.690445          0.847896
  11 │ KNNClassifier               0.805243  1.65854           0.831169
  12 │ RandomForestClassifier      0.805243  2.69173           0.834395
  13 │ PerceptronClassifier        0.692884  6.00335           0.784211
  14 │ DecisionTreeClassifier      0.752809  8.52855           0.78
  15 │ DecisionTreeClassifier      0.404494  36.0437           0.0
  16 │ RandomForestClassifier      0.404494  36.0437           0.0
```

It looks like ```BayesianSubspaceLDA``` is our best candidate: the model offers the lowest ```CrossEntropyLoss```, while maintain comparable ```Accuracy``` and ```F1Score``` with the second-best candidate, ```BayesianLDA```.

The ROC curve can be accessed via calling the object contains it. But first, we can add the legend which indicates which model is it:

```Julia
# Add legend to the plot
axislegend(axis, plot_line, model_names, "Models", position = :rb)

# Inspect the ROC curves
fig
```

![ROC_curve](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Titanic/Plots/ROC_curve.png?raw=true)

As we can see from the plot, the ROC curve for ```BayesianSubspaceLDA``` almost matches one for ```BayesianLDA``` perfectly, which makes sense since both models offer very comparable performance. The difference here is minor.

#### 2.4 Model adjustment

One of the nice thing of ```BayesianSubspaceLDA``` model is that it has **no** hyperparameter to tune. However, the model have the option for us to normalize the training data, which we have already standardized in the beginning. We can confirm that by inspecting the model hyperparameters:

```Julia
# Inspect the hyper parameter space of our model
Bay_subspace_LDA = (@load BayesianSubspaceLDA pkg = MultivariateStats verbosity = true)()
```

```Julia
BayesianSubspaceLDA(
  normalize = false, 
  outdim = 0, 
  priors = nothing)
```

For the benchmark, we use only 70% of our data to train the model and use the rest 30% to evaluate performance. After selecting the best model (with the best combination of hyperparameters if the model has), we can use all of our data to train and evaluate its performance for the last time:

```Julia
# Connect our model with our data
m = machine(Bay_subspace_LDA, features, target)
```

```Julia
untrained Machine; caches model-specific representations of data
  model: BayesianSubspaceLDA(normalize = false, …)
  args:
    1:  Source @138 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @072 ⏎ AbstractVector{OrderedFactor{2}}
```

```Julia
# Train our model using all data
final_model = fit!(m)
```

```Julia
[ Info: Training machine(BayesianSubspaceLDA(normalize = false, …), …).
trained Machine; caches model-specific representations of data
  model: BayesianSubspaceLDA(normalize = false, …)
  args:
    1:  Source @138 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @072 ⏎ AbstractVector{OrderedFactor{2}}

```

```Julia
# Making predictions
target_pre = MLJ.predict(final_model)
```

```Julia
891-element CategoricalDistributions.UnivariateFiniteVector{OrderedFactor{2}, Int64, UInt32, Float64}:
 UnivariateFinite{OrderedFactor{2}}(1=>0.012, 0=>0.988)
 UnivariateFinite{OrderedFactor{2}}(1=>0.0277, 0=>0.972)
 UnivariateFinite{OrderedFactor{2}}(1=>0.0764, 0=>0.924)
 ⋮
 UnivariateFinite{OrderedFactor{2}}(1=>0.822, 0=>0.178)
 UnivariateFinite{OrderedFactor{2}}(1=>0.848, 0=>0.152)
```

```Julia
# Evaluate model performance
DataFrame(Accuracy = accuracy(mode.(target_pre), target), 
        CrossEntropyLoss = mean(cross_entropy(target_pre, target)),
        F1Score = f1score(mode.(target_pre), target))
```

```Julia
1×3 DataFrame
 Row │ Accuracy  CrossEntropyLoss  F1Score  
     │ Float64   Float64           Float64
─────┼──────────────────────────────────────
   1 │ 0.835017          0.421648  0.869565
```

Nothing has changed, in three measures! This suggests that using more data to train does not improve our model performance. The baseline model has reached its limit.

Let's aslo inspect the confusion matrix of our prediction:

```Julia
# Derive the confusion matrix
cm = confusion_matrix(mode.(target_pre), target)
```

```Julia
              ┌───────────────────────────┐
              │       Ground Truth        │
┌─────────────┼─────────────┬─────────────┤
│  Predicted  │      1      │      0      │
├─────────────┼─────────────┼─────────────┤
│      1      │     254     │     59      │
├─────────────┼─────────────┼─────────────┤
│      0      │     88      │     490     │
└─────────────┴─────────────┴─────────────┘
```

So we have correctly predicted 744 cases out of 891 cases, and close to 150 cases were mistakenly classified. It's of course, not a terrible result, but at the same time definitely not a very good one.

#### 2.5. Save the final model

We can save our final model for later use:

```Julia
# Save our machine learning model
MLJ.save("SubSpace_LDA.jls", final_model)
```

### 3. Making prediction on new data 

As we mentioned in the first of the project, the final goal of the project is to predict survivability of passengers on the test set. To use our model to make prediction, we have to repeat all the data processing steps that we did in the second part before fitting it to the model. The saved model can be loaded, unzipped and then used to predict on the data.

#### 3.1 Load, preprocessing new data

The path to the unlabeled data can be found in the project repository:

```Julia
# Path to unlabeled data
path = "test.csv";

# Load the data to our machine
titanic_unlabeled = DataFrame(CSV.File(path));

# Inspect our data
first(titanic_unlabeled, 10)
```

```Julia
10×11 DataFrame
 Row │ PassengerId  Pclass  Name                               Sex      Age       SibSp  Parch  Ticket     Fare      Cabin      Embarked 
     │ Int64        Int64   String                             String7  Float64?  Int64  Int64  String31   Float64?  String15?  String1
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │         892       3  Kelly, Mr. James                   male         34.5      0      0  330911       7.8292  missing    Q
   2 │         893       3  Wilkes, Mrs. James (Ellen Needs)   female       47.0      1      0  363272       7.0     missing    S
   3 │         894       2  Myles, Mr. Thomas Francis          male         62.0      0      0  240276       9.6875  missing    Q
  ⋮  │      ⋮         ⋮                     ⋮                     ⋮        ⋮        ⋮      ⋮        ⋮         ⋮          ⋮         ⋮
   8 │         899       2  Caldwell, Mr. Albert Francis       male         26.0      1      1  248738      29.0     missing    S
   9 │         900       3  Abrahim, Mrs. Joseph (Sophie Hal…  female       18.0      0      0  2657         7.2292  missing    C
  10 │         901       3  Davies, Mr. John Samuel            male         21.0      2      0  A/4 48871   24.15    missing    S
                                                                                                                           4 rows omitted
```

Upon inspection, we can see that our data is a continuation of the ```train.csv``` data, just without the **groundtruth** (target) variable. We will speed up the data preprocessing steps, where we will repeat exactly every steps that we did to our training data in second part, just without detailed explanation:

```Julia
##### Data preprocessing #####
```

```Julia
## -- Feature engineering

# Add FamSize feature
titanic_unlabeled.FamSize = titanic_unlabeled.SibSp .+ titanic_unlabeled.Parch;
```

```Julia
## --  Feature extraction

# Define pattern to extract
pattern = r"\s[MDR](\w+){1,5}\."

# Define, extract and fill in salutation
salutation = String[];

for i in titanic_unlabeled.Name
    ex = match(pattern, i)
    if ex === nothing
        push!(salutation, "Other")
    else 
        push!(salutation, ex.match)
    end
end

# String filtering (optional, for cleaner result)
sal_clean = String[];

for i in salutation
    sal = join(filter!(x -> !(x == '.' || x == ' '), [x for x in i]));
    push!(sal_clean, sal)
end

# Define, count each salutation categories 
words_list = unique(sal_clean)

count_words = Dict{String, Int64}();

for i in sal_clean
    count_words[i] = get(count_words, i, 0) + 1
end

println(count_words)

# Salutation conversion
other = findall(x -> x == "Other", sal_clean);

dona = findall(x -> x == "Dona", sal_clean);

Ms = findall(x -> x == "Ms", sal_clean);

println(titanic_unlabeled.Name[other[1]]);

println(titanic_unlabeled.Name[other[2]]);

println(titanic_unlabeled.Name[dona]);

println(titanic_unlabeled.Name[Ms]);

sal_clean[other[1]] = "Other"

sal_clean[other[2]] = "Other"

sal_clean[dona[1]] = "Other"

sal_clean[Ms[1]] = "Miss"

# Re-inspect adter conversion
unique(sal_clean)

# Add new extracted feature
titanic_unlabeled.Salutation = sal_clean;
```

```Julia
## -- Feature selection

# Drop features with unuseful or overlapped information
titanic_unlabeled = select!(titanic_unlabeled, Not([:PassengerId, :Ticket, :Name, :SibSp, :Parch]));
```

```Julia
## -- Missing data imputation

# Missing data inspection
des = describe(titanic_unlabeled, :nmissing);

des.percentage = des.nmissing ./ nrow(titanic_unlabeled);

des
```

```Julia
8×3 DataFrame
 Row │ variable    nmissing  percentage 
     │ Symbol      Int64     Float64
─────┼──────────────────────────────────
   1 │ Pclass             0  0.0
   2 │ Sex                0  0.0
   3 │ Age               86  0.205742
   4 │ Fare               1  0.00239234
   5 │ Cabin            327  0.782297
   6 │ Embarked           0  0.0
   7 │ FamSize            0  0.0
   8 │ Salutation         0  0.0
```

```Julia
# Drop the Cabin feature since it contains too many missing values (78% in total)
titanic_unlabeled = select!(titanic_unlabeled, Not(:Cabin));

# Missing data imputation
schema(titanic_unlabeled)

titanic_unlabeled = coerce(titanic_unlabeled, 
                          :Pclass => OrderedFactor, 
                          :Sex => Multiclass,
                          :Age => Continuous, 
                          :Fare => Continuous, 
                          :Embarked => Multiclass,
                          :FamSize => Count,
                          :Salutation => Multiclass)

# Load the imputer and imputing mising data
@load FillImputer

filler = machine(FillImputer(), titanic_unlabeled);

fit!(filler);

titanic_unlabeled = MLJ.transform(filler, titanic_unlabeled);

describe(titanic_unlabeled, :nmissing)
```

```Julia
7×2 DataFrame
 Row │ variable    nmissing 
     │ Symbol      Int64
─────┼──────────────────────
   1 │ Pclass             0
   2 │ Sex                0
   3 │ Age                0
   4 │ Fare               0
   5 │ Embarked           0
   6 │ FamSize            0
   7 │ Salutation         0
```

```Julia
## -- Data transformation

# Standardizing and continuous encoding data
@load Standardizer
@load ContinuousEncoder


# Build a transformation pipeline
SimplePipe = Pipeline(
    Standardizer(),
    ContinuousEncoder()
);

trans = machine(SimplePipe, titanic_unlabeled);

MLJ.fit!(trans);

titanic_unlabeled = MLJ.transform(trans, titanic_unlabeled);
```
Our data is ready to be used. Let's unzip our model and load to the working environment:

```Julia
# Unzip the saved model
my_model = machine("SubSpace_LDA.jls")
```

```Julia
trained Machine; caches model-specific representations of data
  model: BayesianSubspaceLDA(normalize = false, …)
  args:
```

We can now use our model to make prediction upon new data:

```Julia
# Making prediction upon unseen data 
results = MLJ.predict_mode(m, titanic_unlabeled)
```

```Julia
418-element CategoricalArrays.CategoricalArray{Int64,1,UInt32}:
 0
 1
 0
 ⋮
 0
 1
```

The result are in binary data (```0``` and ```1```). We can convert them into more readable format:

```Julia
# Convert binary data into more readable format
readable_results = [];

for i in results
    if i == 0
        push!(readable_results, "Not survived")
    else 
        push!(readable_results, "Survived")
    end
end

println(readable_results)
```

```Julia
Any["Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Not survived", "Not survived", "Not survived", "Survived", "Survived", "Survived", "Survived", "Survived", "Not survived", "Survived", "Not survived", "Not survived", "Survived"]
```

### 4. Conclusion

#### 4.1. Some further comments

We have successfully built a machine learning model to predict the survivability of passengers of the Titanic shipwerk. Nevertheless, the performance of our model is just "acceptable", and there are of course rooms for improvement. Below are some ideas to enhance the predictive value of our model even more.

* Since using a single model alone gets us no more than 85% of accuracy, one could try to build network of model and assign weight for them instead;

* The performance evaluation is only correct for the combination of data stable by the seed. One thus could try different algorithms and see whether any measures are improved or not;

* We could go even further with feature engineering and feature extraction, specifically with unused variable such as ```Ticket``` or ```Cabin```, and compare the performance with those that do not use them.

#### 4.2 Conclusion

This section conclude our three-part project. Through out the process, we have done so many steps from data analysis, model selection, model training and testing, performance evaluation, in order to predict the survavivability of the passengers of the Titanic shipwreck based on data in hand. More challenging and sophisticated problems could be the idea of future project. 


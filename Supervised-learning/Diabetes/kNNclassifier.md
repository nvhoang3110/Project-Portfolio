# Build a simple Machine Learning Model to solve a Multiclasses-classification problem 

### Abstract

**The aim of this mini-project**

The purpose of this mini project is to provide a quick demonstration of how to use the Julia programming language as well as MLJ ecosystem to tackle common data science tasks such as Exploratory Data Analysis, Data Visualization and Machine Learning model Building.

### Introduction

**Overview**

![Julia_logo](https://i.imgur.com/zIX4l7G.jpg)

[Julia](https://julialang.org/) is a dynamic, general-purpose programming language capable of high performance scientific computing with high-level code. [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) is a toolbox written in Julia providing a common interface and meta-algorithms for selecting, tuning, evaluating, composing and comparing more than [190 machine learning models](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/#model_list), written in Julia and in other languages. MLJ provides a unify interface to use, compose and tune machine learning models in Julia in a consistent, easy-to-use way, but also extremely flexible and powerful at the same time. This project is a quick demonstration of how to use Julia and MLJ ecosystem to handle various common data science tasks such as Exploratory Data Analysis, Data Visualization and building a Machine Learning model. 

### Data Source

**The data set used in mini-project**

For this mini-project, we are going to use the **diabetes** data set, which descirbes the diabetic condition of a patient based on the level of some chemical compounds found in their blood sample. The [path](Supervised-learning/Diabetes/Data/diabetes.csv) to the ```.csv``` file can be found in the project repository.

### Break down of this mini-project

**Our project is structured as follows:**

1. Installing, importing libraries and data used in our project;

2. Performing EDA and Data Visualization;

3. Building a simple machine learning model, tuning and evaluating model performance;

4. Using our model to make prediction on new data;

5. The last section are some further comments and conclusion.

### 1. Install and load libraries and data for our project

We first start by installing and loading all the libraries used in this project. In Julia this can be done as follows:

```Julia
# Define all packages that we are going to use through out this project
packages = ["CSV", "MLJ", "Distances", "StableRNGs", "PrettyPrinting", "Pipe", "DataFrames", "StatsBase","DataFramesMeta", "CairoMakie"];
```

```Julia
# Install our packages 
import Pkg; Pkg.add(packages);

# Load our packages to the machine
using CSV # To handle CSV file
using MLJ # ML interface
using Distances # Distance function used in k-NN model
using StableRNGs # Simple RNG with stable streams
using PrettyPrinting # For pretty printing
using Pipe # For pipe connection
import DataFrames: DataFrame # Data frame in Julia
import StatsBase: countmap # For counting categorical data
using DataFramesMeta # More data frame manipulation
using CairoMakie # For data visualization
```

For the data, we can load the ```.csv``` file to the environment and convert to standard ```DataFrame``` format:

```Julia
# Provide path to our data. Our data set can be found in project's repository
path = "diabetes.csv";

# Load our data, convert from .csv format to DataFrame
diabetes = DataFrame(CSV.File(path));
```

### 2. Exploratory Data Analysis (EDA) and Data Visualization

In this section, we are going to perform some common data analysis taks: Exploratory Data Analysis and Data Visualization. First, let's take a quick look at our data:

```Julia
# Take a quick look at our data
first(diabetes, 10) |> pprint
```

```Julia
10×4 DataFrame
 Row │ class     glucose  insulin  sspg
     │ String15  Int64    Int64    Int64
─────┼───────────────────────────────────
   1 │ Normal         80      356    124
   2 │ Normal         97      289    117
   3 │ Normal        105      319    143
   4 │ Normal         90      356    199
   5 │ Normal         90      323    240
   6 │ Normal         86      381    157
   7 │ Normal        100      350    221
   8 │ Normal         85      301    186
   9 │ Normal         97      379    142
  10 │ Normal         97      296    131
```

We have a data set that contains in total 145 observations over 4 variables, including 3 continuous features and one categorical variable (the ```class``` column). This idea is confirmed by inspecting the dimensions of our data:

```Julia
# Dimension inspection
num_rows, num_cols = nrow(diabetes), ncol(diabetes)
```

```Julia
(145, 4)
```

The ```class``` columns contains 3 distinct categories indicating the diabetic condition of the patients. It's very easy to see by inspecting all unique entries of the ```class``` column using the ```unique()``` function: 

```Julia
# Confirm that the class column contains categorical data
unique(diabetes.class)
```

```Julia
3-element Vector{String15}:
 "Chemical"
 "Normal"
 "Overt"
```

So to sum up things so far, the data consists of 3 continuous variables and 1 categorical one, with 145 entries in total. The three continuous variables measure level of 3 different chemical compounds in the patients' blood: **insulin level**, **glucose level** and **steady-state plasma glucose level** (sspg), and the only categorical variable ```class``` indicates the diabetic condition based on this measures: non-diabetic (```Normal```), chemically diabetic (```Chemical```) and overtly diabetic (```Overt```). 

To get more insights about the distribution of our data, we can take a look at the statistic summary:

```Julia
# Statistic summary
describe(diabetes, :all) |> pprint
```

```Julia
4×13 DataFrame
 Row │ variable  mean     std      min     q25     median  q75     max    nunique  nmissing  first   last   eltype
     │ Symbol    Union…   Union…   Any     Union…  Union…  Union…  Any    Union…   Int64     Any     Any    DataType
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ class                       Normal                          Overt  3               0  Normal  Overt  CategoricalValue{String15, UInt3…
   2 │ glucose   121.986  63.9304  70      90.0    97.0    112.0   353                    0  80      346    Int64
   3 │ insulin   540.786  319.565  45      352.0   403.0   558.0   1568                   0  356     1568   Int64
   4 │ sspg      186.117  120.935  10      118.0   156.0   221.0   748                    0  124     15     Int64
```

Here are some observations:
* All three continuous variables, ```glucose```, ```sspg``` and ```sspg```, might be measured by the same unit, since they have a relatively similar mean, quantiles. While ```sspg``` varies in a much wider range, the difference is not to drastic;

* ```insulin``` variable has the largest mean and standard deviation, but its smallest value is comparable with the one of ```glucose``` or ```sspg``` variable;

* Again, our only category variable ```class``` has three distinct classes: ```Normal```, ```Chemical``` and ```Overt```;

* Also, there's no missing data in our data set, which is a good thing. In reality most of the time, it's not the case.

Since our target variable is of type multi-classes, we can inspect its distribution to detect any imbalanced in classes. This can be done by ploting the total number of entries for each class. 

The first step is to filter our data with respect to the classes. In other word, we filter our data using the classes in the ```class``` column:

```Julia
# Filter our data based on classes
normal = filter(:class => x -> x == "Normal", diabetes);

chemical = filter(:class => x -> x == "Chemical", diabetes);

overt = filter(:class => x -> x == "Overt", diabetes);
```

Also for the sake of convenient and reusable code later, we will define some global styling, format and indexing for our plot:

```Julia
## -- Define some global styling, format and indexing 

# Plot font
plot_font = "TeX Gyre Heros Makie Bold";

# Tick format
tick_format = v -> format.(v, commas = true);

# Color palette
color_con = Dict(normal => "#FC7808", chemical => "#8C00EC", overt => "#107A78");

# Indexing
dat_index = Dict(normal => 1, chemical => 2, overt => 3);
```

Let's plot the distribution of our categorical variable (```class```) using a **BarPlot**:

```Julia
## -- Barplot 

bar_plot = begin

    # Reset all modification to default
    set_theme!()

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080), fontsize = 30)
    
    # Initialize axis
    axis = fig[1, 1] = Axis(fig, 
                            title = "", 
                            xlabel = "Diabetic status", 
                            xticks = (1:3, ["Normal", "Chemical", "Overt"]),
                            xlabelfont = "TeX Gyre Heros Makie Bold",
                            ylabel = "Count",
                            ylabelfont = "TeX Gyre Heros Makie Bold")
    
    # Plot
    for i in [normal, chemical, overt]
        barplot!(axis, dat_index[i], nrow(i), color = color_con[i])
    end
    
    # Return the final plot
    fig
end
```

![bar_plot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Diabetes/Plots/bar_plot.png?raw=true)

From the plot, we can see that our initial guess is correct: a clear imbalanced in our categorical variable. There are more paients with ```Normal``` diabetic condition than both ```Chemical``` and ```Overt``` diabetic conditions. In fact, the nummber of patients with ```Normal``` diabetic condition is almost equal to the combined number of both ```Chemical``` and ```Overt``` diabetic conditions:

```Julia
# Count the number of patients with normal diabetic condition vs the rest
countmap(diabetes.class)
```

```Julia
Dict{CategoricalValue{String15, UInt32}, Int64} with 3 entries:
  "Overt"    => 33
  "Chemical" => 36
  "Normal"   => 76
```

So it's clear that in our data, class ```normal``` has the most number of entries: in total there are 76 entries versus 69 entries with some sign of diabetes. Imbalanced data often causes troubles,however for our particular situation, since the size of our data set is quite small (145 entries over 4 variables), and the imbalanced is not that significant (something likes 90% of the data belongs to one class and the rest is for other classes), there's no need any form of specific data treatment.

For the continuos variables, to inspect how they interact and are related, we can plot them against each other, colored by classes (diabetic condition of the patients). We will draw all three combinations of them, and put in a single scene for easy comparison:

```Julia
## -- Scatter plot

scatter_plot = begin

    # Reset all modification to default
    set_theme!()

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080))

    # Update theme and font size
    update_theme!(with_theme = theme_light(), fontsize = 30)

    # Initialize axis
    axis1 = fig[1, 1] = Axis(fig, title = "",
                            xlabel = "SSPG",
                            xlabelfont = plot_font,
                            ylabel = "Insulin",
                            ylabelfont = plot_font,
                            ytickformat = tick_format)

    axis2 = fig[1, 2] = Axis(fig, title = "",
                            xlabel = "Glucose",
                            xlabelfont = plot_font,
                            ylabel = "Insulin",
                            ylabelfont = plot_font,
                            ytickformat = tick_format)

    axis3 = fig[2, 1] = Axis(fig, title = "",
                            xlabel = "SSPG",
                            xlabelfont = plot_font,
                            ylabel = "Glucose",
                            ylabelfont = plot_font,
                            ytickformat = tick_format)

    # Plot
    the_plot = []

    for i in [normal, chemical, overt]
        current_plot = scatter!(axis1, i.sspg, i.insulin, 
                color = (color_con[i], 0.65), 
                markersize = 20)
        push!(the_plot, current_plot)
        
        scatter!(axis2, i.glucose, i.insulin, 
                color = (color_con[i], 0.65), 
                markersize = 20)
        
        scatter!(axis3, i.sspg, i.glucose, 
                color = (color_con[i], 0.65), 
                markersize = 20)
    end

    # Add legend
    leg = fig[2, 2] = Legend(fig, the_plot, ["Normal", "Chemical", "Overt"], 
                            "Diabetic status", framevisible = false,
                            tellheight = false, tellwidth = false)
    
    # Return the final plot
    fig
end
```

![Scatter_plot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Diabetes/Plots/scatter_plot.png?raw=true)

If you prefer a 3D-plot of our continuous variables, to combine all 3 features together, we can certainly do it. Although it is not not recommended, since the human eye reads 3D figure really bad, which may lead to lack of accuracy and poor interpretation about the data:

```Julia
plot_3D = begin
    
    # Reset all modification to default
    set_theme!()

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080))

    # Update theme and font size
    update_theme!(with_theme = theme_light(), fontsize = 30)

    # Initialize axis
    axis = fig[1, 1] = Axis3(fig, title = "",
                            xlabel = "SSPG",
                            viewmode = :fitzoom,
                            xlabeloffset = 50,
                            xlabelfont = "TeX Gyre Heros Makie Bold",
                            ylabel = "Glucose",
                            ylabeloffset = 70,
                            ylabelfont = "TeX Gyre Heros Makie Bold",
                            zlabel = "Insulin",
                            zlabeloffset = 105,
                            zlabelfont = "TeX Gyre Heros Makie Bold",
                            ztickformat = v -> format.(v, commas = true)
                            )

    # Plot 
    the_plot = []

    for i in [normal, chemical, overt]
        current_plot = scatter!(axis, i.sspg, i.glucose, i.insulin, 
                            color = (color_con[i], 0.65), markersize = 20)
        push!(the_plot, current_plot)
    end

    # Add legend
    leg = fig[1, 1] = Legend(fig, the_plot, ["Normal", "Chemical", "Overt"], orientation = :vertical,
                            "Diabetic status", framevisible = false, colgap = 20, tellwidth = false, tellheight = false,
                            halign = :right, valign = :center, labelsize = 30, titlesize = 30)

    # Return the final plot
    fig

end
```

![3D_plot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Diabetes/Plots/3D_plot.png?raw=true)

Here are some intuitions that we get from our plots:

* Patients with ```Overt``` diabetic condition expose a very  high level of both ```insulin``` and ```glucose``` in their blood, while the ```sspg``` level shows a slight lower than the other two. This evidence may suggest that it is not a good feature to separate the classes;

* Patients with ```chemical``` diabetic condition show a "somewhat" higher level of both ```insulin``` and ```sspg``` in their blood compared to patients with ```Normal``` diabetic condition. Their ```sspg``` level also varies in a wider range, though the difference is not too dramatic and a large part of them overlaps with each other.

In conclusion, we can see a clear difference in the data of patient with ```Overt``` diabetic condition compared to the rest. It is also harder to separate patients with ```Chemical``` diabetic condition with patients with ```Normal``` diabetic condition, since their data are very similar, but of course not the same. 

In the next section, we are going to build a machine learning model to predict the diabetic condition of patients based on these data.

### 3. Building a machine learning model

#### 3.1. Data preprocessing

Our first step when it comes to building a machine model using MLJ is to coerce all machine types into scientific types.

* **Machine type** refers to the Julia type being use to represent the object within a machine (for example, Int64 or Float64);

* **Scientific type** is one of the types defined in the package [ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl), reflecting how the object should be interpreted (for example, ```Continous``` or ```Multiclass```).

Data coercion is a crucial step, since models in MLJ articulate their data requirements using scientific types. More on scientific types in MLJ can be found in [ScientificTypes.jl](https://juliaai.github.io/ScientificTypes.jl/dev/) documentation. 

To inspect the current types of our data, we use the ```schema()``` function:

```Julia
# Inspect the current machine type and scientific type of our data
schema(diabetes)
```

```Julia
┌─────────┬──────────┬──────────┐
│ names   │ scitypes │ types    │
├─────────┼──────────┼──────────┤
│ class   │ Textual  │ String15 │
│ glucose │ Count    │ Int64    │
│ insulin │ Count    │ Int64    │
│ sspg    │ Count    │ Int64    │
└─────────┴──────────┴──────────┘
```

So the 3 continuous variables are stored in the computer as ```Int64```. The only categorical variable is of type ```String15```. From the documentation, we know that we should convert the ```class``` variable from ```String15``` to ```OrderedFactor```, and the other three continuous variables to ```Continuous```. This can be done using the ```coerce()``` function:

```Julia
# Convert data from machine type to scientific type
diabetes = coerce(diabetes, 
            :class => OrderedFactor, 
            :glucose => Continuous, 
            :insulin => Continuous, 
            :sspg => Continuous);
```

Let's double-check the type of our data after converting to scientific type, as well as inspect the how each class are ordered in the column:

```Julia
# Inspect types of our data after coercion
schema(diabetes)
```

```Julia
┌─────────┬──────────────────┬────────────────────────────────────┐
│ names   │ scitypes         │ types                              │
├─────────┼──────────────────┼────────────────────────────────────┤
│ class   │ OrderedFactor{3} │ CategoricalValue{String15, UInt32} │
│ glucose │ Continuous       │ Float64                            │
│ insulin │ Continuous       │ Float64                            │
│ sspg    │ Continuous       │ Float64                            │
└─────────┴──────────────────┴────────────────────────────────────┘
```

```Julia
# Inspect the level of our categirical data
levels(diabetes.class)
```

```Julia
3-element Vector{String15}:
 "Chemical"
 "Normal"
 "Overt"
```

So the scientific types of our data are all proper. We just need to reorder the levels of them based on how sereve the diabetic condition of a patient is:

```Julia
# Reorder the levels of the categorical data
levels!(diabetes.class, ["Normal", "Chemical", "Overt"]);
```

```Julia
# Double check the level after reordering
levels(diabetes.class)
```

```Julia
3-element Vector{String15}:
 "Normal"
 "Chemical"
 "Overt"
```

Everything is correct now. The next step is to split our data accordingly into ```target``` and ```features``` variable. 

* ```target``` variable is the varialbe in the data set that we're trying predict; 

* ```features``` variable contains the predictors, which are variables we hope to contain the information needed for the algorithm to make predictions.

For our instance, the ```target``` variable is the ```class``` column, wherase the feature space are the other 3 continuous variable, ```glucose```, ```insulin``` and ```sspg```. Let's split our data vertically by this direction, and shuffle it along the process:

```Julia
# Set seed for reproduceable code
rng = StableRNG(325);

# Split the data into target and feature varialbes.Shuffle along the process
target, features = unpack(diabetes, ==(:class), shuffle = true, rng = rng)
```

```Julia
(CategoricalArrays.CategoricalValue{String15, UInt32}["Normal", "Normal", "Normal", "Normal", "Chemical", "Overt", "Overt", "Normal", "Overt", "Overt"  …  "Overt", "Overt", "Chemical", "Chemical", "Normal", "Normal", "Normal", "Chemical", "Overt", "Normal"], 145×3 DataFrame
 Row │ glucose  insulin  sspg    
     │ Float64  Float64  Float64
─────┼───────────────────────────
   1 │    90.0    323.0    240.0
   2 │    78.0    290.0    136.0
  ⋮  │    ⋮        ⋮        ⋮
 144 │   146.0    847.0    103.0
 145 │    91.0    353.0    221.0
                 141 rows omitted)
```

To enhance the data preprocessing step even further, we can even further inspect the histograms of our features, for example, to see whether there's enough variation among our continuous data or not. For this project, we simply skip this it and jump straight to the model-building step.

#### 3.2. Machine learning model

Choosing a proper machine model for a specific data set is not an easy task. Since MLJ has a model registry, we can search for models based on their properties without loading all the packages containing the model. For our case, let's look at all of the models that fit our target variable and feature space, return a probabilistic result and are written in just **Julia**:

```Julia
# Model searching with criterions
models() do model
    matching(model, features, target) &&
    model.prediction_type == :probabilistic &&
    model.is_pure_julia
end
```

```Julia
19-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
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
 (name = LogisticClassifier, package_name = MLJLinearModels, ... )
 (name = MultinomialClassifier, package_name = MLJLinearModels, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = DecisionTree, ... )
 (name = SubspaceLDA, package_name = MultivariateStats, ... )
```

There are in total 19 models that meet our requirements. For such a small data set, selecting a sophisticated algorithm, such as **Support Vector Machine**, could easily result in an overfitting model. For that reason, we will pick a fairly simple and computational inexpensive to train model, **k-Nearest Neighbor** from the ```DecisionTreeClassifier``` package to build our classifier (we have to specify which package we want to import our model since there are more than just one package providing the same thing):

```Julia
# Load our k-NN classifier from the NearestNeighborModels package
kNN = (@load KNNClassifier pkg = NearestNeighborModels verbosity = 0)()
```

```Julia
KNNClassifier(
  K = 5,
  algorithm = :kdtree, 
  metric = Euclidean(0.0), 
  leafsize = 10,
  reorder = true, 
  weights = Uniform())
```

The default measure to calculate distances is the ```Euclidean``` distance. We can now wrap the model with our data to create a machine which will store the outcome. In MLJ, a ```machine``` bines a model (i.e the choice of algorithm and its hyperparameter) to its data. It's also the object in which **learned parameters** are stored: 

```Julia
# Wrap a machine around the data (target and features)
knn = machine(kNN, features, target)
```

```Julia
untrained Machine; caches model-specific representations of data
  model: KNNClassifier(K = 5, …)
  args:
    1:  Source @422 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @008 ⏎ AbstractVector{OrderedFactor{3}}
```

A common training / testing strategy is to split the data into **training** and **testing** set, training data for training and testing data for performance evaluation. We can choose the common ratio of 70% data for training and the rest 30% for testing, then split the data based on index. We also shuffle the data in the process:

```Julia
# Split our data into based on index into training and testing
train, test = partition(collect(eachindex(target)), 0.7, shuffle = true, rng = rng)
```

```Julia
([26, 48, 92, 76, 24, 119, 16, 44, 17, 133  …  7, 97, 52, 10, 125, 106, 93, 137, 63, 29], [2, 122, 85, 35, 141, 98, 102, 8, 41, 86  …  139, 100, 79, 56, 81, 37, 55, 144, 73, 136])
```

We can now fit our the data on training set, making predictions and then eveluate the performance on testing set. We are going to use various measures to evaluate model performance for multi-classes classification task: ```Accuracy```, ```Cross-Entropy Loss``` and ```F1-Score```:

```Julia
# Fit our model on training set
fit!(knn, rows = train)
```

```Julia
[ Info: Training machine(KNNClassifier(K = 5, …), …).
trained Machine; caches model-specific representations of data
  model: KNNClassifier(K = 5, …)
  args:
    1:  Source @422 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @008 ⏎ AbstractVector{OrderedFactor{3}}
```

```Julia
# Making prediction on the test set
target_pre = predict(knn, rows = test)
```

```Julia
43-element CategoricalDistributions.UnivariateFiniteVector{OrderedFactor{3}, String15, UInt32, Float64}:
 UnivariateFinite{OrderedFactor{3}}(Normal=>1.0, Chemical=>0.0, Overt=>0.0)
 UnivariateFinite{OrderedFactor{3}}(Normal=>0.0, Chemical=>0.0, Overt=>1.0)
 ⋮
 UnivariateFinite{OrderedFactor{3}}(Normal=>0.0, Chemical=>0.0, Overt=>1.0)
 UnivariateFinite{OrderedFactor{3}}(Normal=>0.0, Chemical=>0.0, Overt=>1.0)
```

```Julia
# Evaluate model performance: Accuracy, Cross-Entropy Loss and F1-Score
DataFrame(
    CE_Loss = mean(cross_entropy(target_pre, target[test])),
    Accuracy = accuracy(mode.(target_pre), target[test]),
    F1_score = multiclass_f1score(mode.(target_pre), target[test])
)
```

```Julia
1×3 DataFrame
 Row │ CE_Loss   Accuracy  F1_score 
     │ Float64   Float64   Float64
─────┼──────────────────────────────
   1 │ 0.136443  0.930233  0.934199
```

Not to bad for our first try. With a simple model such as k-NN using default hyperparameter, we're able to achieve accuracy around ```0.9302``` and F1-score of ```0.9341``` also, quite impressive! Let's also look at the confusion matrix of our predictions to see how many cases were correctly and incorrectly classified.

```Julia
# Derive the confusion matrix
cm = confusion_matrix(mode.(target_pre), target[test])
```

```Julia
              ┌─────────────────────────────────────────┐
              │              Ground Truth               │
┌─────────────┼─────────────┬─────────────┬─────────────┤
│  Predicted  │   Normal    │  Chemical   │    Overt    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│   Normal    │     26      │      2      │      0      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│  Chemical   │      1      │      9      │      0      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│    Overt    │      0      │      0      │      5      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

So our model was able to correctly predict all patients with ```Over``` diabetic condition. There are just two cases of ```Normal``` diabetic condition and one case of ```Chemical``` diabetic condition that are mistakenly classified as the other, which is not a lot. Our model did a really good job to separate the classes in the data.

#### 3.3. Model tuning

To improve our model performance even further, we can tune it a little bit. For our k-NN classifier, we can tune the hyperparameter ```K``` - the number of nearest neighbor around each data point, and also how the distance between each data point is measured. In this example, we only tune the value of ```K``` and see how it affects the overall performance. 

First we need to set the range of values for our hyperparameter ```K```. The default value of ```K``` is 5, so we will let ```K``` varies from 1 to 20:

```Julia
# Set range for our hyperparameter k
k = range(kNN, :K, lower = 1, upper = 20)
```

```Julia
NumericRange(1 ≤ K ≤ 20; origin=10.5, unit=9.5)
```

For our tuned our model, we use grid search strategy, set the resolution to 20 to match with the number of ```K``` we are going to tune (resulting in a 20 x 20 grid) and resample using **cross-validation** strategy with 10 folds. We will pick ```Accuracy``` as the measure that the tuning process will try to maximize:

```Julia
# Define the tunning strategy for our model
tm = TunedModel(model = kNN,
                tuning = Grid(resolution = 20), 
                resampling = CV(nfolds = 10, rng = rng),
                ranges = k,
                measure = Accuracy())
```

```Julia
ProbabilisticTunedModel(
  model = KNNClassifier(
        K = 5, 
        algorithm = :kdtree,
        metric = Euclidean(0.0),
        leafsize = 10,
        reorder = true,
        weights = NearestNeighborModels.Uniform()),
  tuning = Grid(
        goal = nothing,
        resolution = 20,
        shuffle = true,
        rng = Random._GLOBAL_RNG()), 
  resampling = CV(
        nfolds = 10,
        shuffle = true,
        rng = StableRNGs.LehmerRNG(state=0x4ed30981c957ea81d8531c7eb09b01cb)), 
  measure = Accuracy(), 
  weights = nothing,
  class_weights = nothing,
  operation = nothing,
  range = NumericRange(1 ≤ K ≤ 20; origin=10.5, unit=9.5),
  selection_heuristic = MLJTuning.NaiveSelection(nothing), 
  train_best = true,
  repeats = 1,
  n = nothing,
  acceleration = CPU1{Nothing}(nothing), 
  acceleration_resampling = CPU1{Nothing}(nothing),
  check_measure = true,
  cache = true)
```

Let's wrap our model into a machine along with our data again:

```Julia
# Wrap our tuned model
knn_tuned = machine(tm, features, target)
```

```Julia
untrained Machine; does not cache data
  model: ProbabilisticTunedModel(model = KNNClassifier(K = 5, …), …)
  args:
    1:  Source @056 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @205 ⏎ AbstractVector{OrderedFactor{3}}
```

And then we can fit our model and evaluate the performance on testing data one more time, just like before:

```Julia
# Fit our model to training data
fit!(knn_tuned, rows = train)
```

```Julia
[ Info: Training machine(ProbabilisticTunedModel(model = KNNClassifier(K = 5, …), …), …).
[ Info: Attempting to evaluate 20 models.
Evaluating over 20 metamodels: 100%[=========================] Time: 0:00:09
trained Machine; does not cache data
  model: ProbabilisticTunedModel(model = KNNClassifier(K = 5, …), …)
  args:
    1:  Source @056 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @205 ⏎ AbstractVector{OrderedFactor{3}}
```

```Julia
# Making prediction on testing data
target_pre_tuned = MLJ.predict(knn_tuned, rows = test)
```

```Julia
43-element CategoricalDistributions.UnivariateFiniteVector{OrderedFactor{3}, String15, UInt32, Float64}:
 UnivariateFinite{OrderedFactor{3}}(Normal=>1.0, Chemical=>0.0, Overt=>0.0)
 UnivariateFinite{OrderedFactor{3}}(Normal=>0.0, Chemical=>0.0, Overt=>1.0)
 ⋮
 UnivariateFinite{OrderedFactor{3}}(Normal=>0.0, Chemical=>0.0, Overt=>1.0)
 UnivariateFinite{OrderedFactor{3}}(Normal=>0.0, Chemical=>0.0, Overt=>1.0)
```

```Julia
# Performance evaluation
DataFrame(
    CE_Loss = mean(cross_entropy(target_pre_tuned, target[test])),
    Accuracy = accuracy(mode.(target_pre_tuned), target[test]),
    F1_score = multiclass_f1score(mode.(target_pre_tuned), target[test])
)
```

```Julia
1×3 DataFrame
 Row │ CE_Loss   Accuracy  F1_score 
     │ Float64   Float64   Float64
─────┼──────────────────────────────
   1 │ 0.143981  0.953488  0.957351
```

So after the tuning process, the ```Accuracy``` and ```F1-Score``` of our prediction on the test set do improve a little bit, from around ```0.93``` each to ```0.95```. However, ```Cross-Entropy Loss``` does suffer, from around ```0.13``` to ```0.14```. It's because we prioritize to maximize ```Accuracy``` in the tuning process, not the other way around. Since we can raise both ```Accuracy``` and ```F1-Score``` , the trade off is acceptable.

Let's look at the confusion matrix of our prediction again after tuning again:

```Julia
# Compute the confusion matrix of our prediction on testing data
cm_tuned = confusion_matrix(mode.(target_pre_tuned), target[test])
```

```Julia
              ┌─────────────────────────────────────────┐
              │              Ground Truth               │
┌─────────────┼─────────────┬─────────────┬─────────────┤
│  Predicted  │   Normal    │  Chemical   │    Overt    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│   Normal    │     26      │      1      │      0      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│  Chemical   │      1      │     10      │      0      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│    Overt    │      0      │      0      │      5      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

So we can see the improvement being made here after tuning. The algorithm is still able to classify all patients with ```Overt``` diabetic condition correctly, while reduces one misclassification case, from the total of 3 to the total of 2. Just a minor improvement, perhaps because our simple algorithm has reach its limit.

#### 3.4. Reporting results

Let's inspect some more information about our tuned model. First we take a look at the set of tuned hyperparameters and specifically the number of optimal nearest neighbor:

```Julia
# Retrieve the best model information
best_knn = fitted_params(knn_tuned).best_model;
```

```Julia
# Inspect the best value of k
@show best_knn.K
```

```Julia
best_knn.K = 7
7
```

So the best value of ```K``` in our tuned model is 7, which is higher than the default setting of 5. To get a complete report information of the tuning process, we can use the ```report``` method:

```Julia
# Get the full report for the tuning process
r = report(knn_tuned);
```

```Julia
(best_model = KNNClassifier(K = 7, …),
 best_history_entry = (model = KNNClassifier(K = 7, …),
                       measure = [Accuracy()],
                       measurement = [0.8927272727272728],
                       per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.9, 1.0, 0.7, 0.9, 1.0, 0.9, 0.9]],),
 history = NamedTuple{(:model, :measure, :measurement, :per_fold), Tuple{NearestNeighborModels.KNNClassifier, Vector{Accuracy}, Vector{Float64}, Vector{Vector{Float64}}}}[(model = KNNClassifier(K = 17, …), measure = [Accuracy()], measurement = [0.8627272727272727], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 0.9, 0.7, 1.0, 1.0, 0.9, 0.7]]), (model = KNNClassifier(K = 6, …), measure = [Accuracy()], measurement = [0.8827272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.9, 0.9, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 9, …), measure = [Accuracy()], measurement = [0.8736363636363637], per_fold = [[0.8181818181818181, 0.8181818181818181, 0.9, 0.8, 1.0, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 15, …), measure = [Accuracy()], measurement = [0.8727272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 0.9, 0.7, 1.0, 1.0, 0.9, 0.8]]), (model = KNNClassifier(K = 1, …), measure = [Accuracy()], measurement = [0.8645454545454545], per_fold = [[0.7272727272727273, 0.8181818181818181, 0.9, 0.8, 1.0, 0.6, 1.0, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 16, …), measure = [Accuracy()], measurement = [0.8727272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 0.9, 0.7, 1.0, 1.0, 0.9, 0.8]]), (model = KNNClassifier(K = 20, …), measure = [Accuracy()], measurement = [0.8336363636363637], per_fold = [[0.8181818181818181, 0.8181818181818181, 0.8, 0.8, 0.9, 0.7, 1.0, 0.9, 0.9, 0.7]]), (model = KNNClassifier(K = 7, …), measure = [Accuracy()], measurement = [0.8927272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.9, 1.0, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 5, …), measure = [Accuracy()], measurement = [0.8636363636363636], per_fold = [[0.8181818181818181, 0.8181818181818181, 0.9, 0.9, 0.8, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 4, …), measure = [Accuracy()], measurement = [0.8727272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.9, 0.8, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 11, …), measure = [Accuracy()], measurement = [0.8836363636363636], per_fold = [[0.8181818181818181, 0.8181818181818181, 0.9, 0.9, 1.0, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 3, …), measure = [Accuracy()], measurement = [0.8536363636363637], per_fold = [[0.9090909090909091, 0.7272727272727273, 0.9, 0.9, 0.8, 0.7, 0.9, 1.0, 0.8, 0.9]]), (model = KNNClassifier(K = 10, …), measure = [Accuracy()], measurement = [0.8736363636363637], per_fold = [[0.8181818181818181, 0.8181818181818181, 0.9, 0.8, 1.0, 0.7, 1.0, 0.9, 0.9, 0.9]]), (model = KNNClassifier(K = 18, …), measure = [Accuracy()], measurement = [0.8527272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 0.9, 0.7, 1.0, 0.9, 0.9, 0.7]]), (model = KNNClassifier(K = 13, …), measure = [Accuracy()], measurement = [0.8927272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 1.0, 0.7, 1.0, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 14, …), measure = [Accuracy()], measurement = [0.8627272727272727], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 0.9, 0.7, 1.0, 1.0, 0.9, 0.7]]), (model = KNNClassifier(K = 8, …), measure = [Accuracy()], measurement = [0.8827272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 1.0, 0.7, 0.9, 1.0, 0.9, 0.9]]), (model = KNNClassifier(K = 19, …), measure = [Accuracy()], measurement = [0.8527272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 0.9, 0.7, 1.0, 0.9, 0.9, 0.7]]), (model = KNNClassifier(K = 2, …), measure = [Accuracy()], measurement = [0.8536363636363637], per_fold = [[0.8181818181818181, 0.8181818181818181, 0.8, 0.9, 0.9, 0.7, 1.0, 0.9, 0.8, 0.9]]), (model = KNNClassifier(K = 12, …), measure = [Accuracy()], measurement = [0.8927272727272728], per_fold = [[0.9090909090909091, 0.8181818181818181, 0.9, 0.8, 1.0, 0.7, 1.0, 1.0, 0.9, 0.9]])],
 best_report = (),
 plotting = (parameter_names = ["K"],
             parameter_scales = [:linear],
             parameter_values = Any[17; 6; … ; 2; 12;;],
             measurements = [0.8627272727272727, 0.8827272727272728, 0.8736363636363637, 0.8727272727272728, 0.8645454545454545, 0.8727272727272728, 0.8336363636363637, 0.8927272727272728, 0.8636363636363636, 0.8727272727272728, 0.8836363636363636, 0.8536363636363637, 0.8736363636363637, 0.8527272727272728, 0.8927272727272728, 0.8627272727272727, 0.8827272727272728, 0.8527272727272728, 0.8536363636363637, 0.8927272727272728],),)
```

Well, that is quite a lot of information! But of course, we don't need all of them. Everything we need is the numbers of nearest neighbor and their associated average accuracy of the model tested on the cross-validation strategy. This information can by found in the field ```plotting```, so let's retrieve it and visualize to see.

```Julia
res = r.plotting
```

```Julia
(parameter_names = ["K"],
 parameter_scales = [:linear],
 parameter_values = Any[17; 6; … ; 2; 12;;],
 measurements = [0.8627272727272727, 0.8827272727272728, 0.8736363636363637, 0.8727272727272728, 0.8645454545454545, 0.8727272727272728, 0.8336363636363637, 0.8927272727272728, 0.8636363636363636, 0.8727272727272728, 0.8836363636363636, 0.8536363636363637, 0.8736363636363637, 0.8527272727272728, 0.8927272727272728, 0.8627272727272727, 0.8827272727272728, 0.8527272727272728, 0.8536363636363637, 0.8927272727272728],)
```

That's still more than what we actually need. We can create a ```DataFrame``` to select just the information of ```K``` and the ```Accuracy```. The values of ```K``` are stored in a ```Matrix{Any}```, so it's better to transform them into a vector and convert their type to ```Int64```:

```Julia
# Retrieve our plotting data, transform into DataFrame. Convert the type of K
dat = DataFrame(K = convert(Array{Int64}, vec(res.parameter_values)), 
              Accuracy = res.measurements)
```

```Julia
20×2 DataFrame
 Row │ K      Accuracy 
     │ Int64  Float64
─────┼─────────────────
   1 │    17  0.862727
   2 │     6  0.882727
  ⋮  │   ⋮       ⋮
  19 │     2  0.853636
  20 │    12  0.892727
        16 rows omitted
```

Unfortunately, the values of ```K``` in our ```Dataframe``` appear in no particular order, so we have to sort them by row:

```Julia
# Define the sort order: the value of K increased by one unit
the_order = sort(dat.K);

# Sort the data based on the value of K
dat_sort = @rorderby dat findfirst(==(:K), the_order)
```

```Julia
20×2 DataFrame
 Row │ K      Accuracy 
     │ Int64  Float64
─────┼─────────────────
   1 │     1  0.864545
   2 │     2  0.853636
  ⋮  │   ⋮       ⋮
  19 │    19  0.852727
  20 │    20  0.833636
        16 rows omitted
```

Much better now. Finally we are ready to plot the result:

```Julia
# Visualize how different value of K affects Accuracy of the model
line_scatter_plot = begin
    
    # Reset all modification to default
    set_theme!()

    # Initialize empty scene
    fig = Figure(resolution = (1920, 1080))

    # Start by setting theme, layout for our plot 
    update_theme!(fontsize = 30, markersize = 25);

    # Initialize axis
    axis = fig[1, 1] = Axis(fig, title = "", 
                        xlabel = "Number of nearest neighbor - K", 
                        xlabelfont = "TeX Gyre Heros Makie Bold",
                        ylabel = "Accuracy",
                        ylabelfont = "TeX Gyre Heros Makie Bold")

    # Plot 
    scatterlines!(axis, dat_sort.K, dat_sort.Accuracy, color = "#034f84", 
            markercolor = "#c94c4c", markersize = 18, linewidth = 4.5)

    # Return the final plot
    fig

end
```

![line_scatter_plot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Diabetes/Plots/line_scatter_plot.png?raw=true)

From the plot, we can see that the ```Accuracy``` reaches its peak value at ```K``` equals to 7, 12 and 13. Since setting the number of nearest neighbor to 7 results in a similar performance as setting it to 12 or 13, the algorithm thus selected that value as the optimal one. Anything more than 7 would end up in similar or even worse performance, with the additional cost of computational budget.

#### 3.5. Assemble the final model

As we are happy with the performance of our model, let's retrieve the best combination of hyperparameters, and build a final model using all of our data at hand and save it as our diabetic classifier. 

The best combination of hyperparameters is saved in an object ```best_knn``` before, so let's retrieve them:

```Julia
# Inspect the best model
best_knn
```

```Julia
KNNClassifier(
  K = 7, 
  algorithm = :kdtree,
  metric = Euclidean(0.0),
  leafsize = 10,
  reorder = true,
  weights = NearestNeighborModels.Uniform())
```

Not much of a difference from the baseline mode, only the value of ```K``` is set to 7. Using this information, we can compose our final model:

```Julia
# Compose the final model
final_model = final_model = (@load KNNClassifier pkg = NearestNeighborModels verbosity = 0)(K = 7);
```

Now let's use all of our data to train it:

```Julia
# Connect the model with data using machine
knn_final = machine(final_model, features, target);
```

```Julia
untrained Machine; caches model-specific representations of data
  model: KNNClassifier(K = 7, …)
  args:
    1:  Source @303 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @249 ⏎ AbstractVector{OrderedFactor{3}}
```

```Julia
# Train the model with all the data
fit!(knn_final);

knn_final
```

```Julia
trained Machine; caches model-specific representations of data
  model: KNNClassifier(K = 7, …)
  args:
    1:  Source @303 ⏎ Table{AbstractVector{Continuous}}
    2:  Source @249 ⏎ AbstractVector{OrderedFactor{3}}
```

Done! Our model can be saved for later prediction on new data:

```Julia
# Save the final model
MLJ.save("knn_classifier.jls", knn_final)
```

### 4. Using the model to make new prediction

We have our model ready to be used, to classify the diabetic condition of new, unseen patients. Imagine their blood testing data is as follows:

```Julia
# New data from our patients
diabetes_new = DataFrame(glucose = [82, 108, 300],
                            insulin = [361, 288, 1052], 
                            sspg = [200, 186, 135])
```

```Julia
3×3 DataFrame
 Row │ glucose  insulin  sspg  
     │ Int64    Int64    Int64
─────┼─────────────────────────
   1 │      82      361    200
   2 │     108      288    186
   3 │     300     1052    135
```

We can predict the diabetic condition of these patients based on their data using our model. But first, let's not forget to transform this raw data to proper form before feeding to the model. In this case, we only need to coerce the machine type of each column to a proper scientific types as before:

```Julia
# Coerce new data types to proper scientific types
diabetes_new = coerce(diabetes_new, :glucose => Continuous, 
                                   :insulin => Continuous, 
                                   :sspg => Continuous)
```
```Julia
Row │ glucose  insulin  sspg    
     │ Float64  Float64  Float64
─────┼───────────────────────────
   1 │    82.0    361.0    200.0
   2 │   108.0    288.0    186.0
   3 │   300.0   1052.0    135.0
```

The data is now ready to parse to our model to make prediction. To do that, we first invoke our saved model from a ```.jls``` file, deserialize it and then make predictions:

```Julia
## -- Making predictions on new, unseen data

# Provide the path to our saved model
path = "knn_classifier.jls";

# Call our saved model and deserialize
my_model = machine(path)
```

```Julia
trained Machine; caches model-specific representations of data
  model: KNNClassifier(K = 7, …)
  args: 
```

```Julia
# Make predictions on new data
MLJ.predict_mode(my_model, diabetes_new)
```

```Julia
3-element CategoricalArray{String15,1,UInt32}:
 "Normal"
 "Normal"
 "Overt"
```

Our model predicts the first two patients have ```Normal``` diabetic condition, while the third one has ```Overt``` diabetic contion. 

### 5. Conclusion

#### 5.1 Some further comments

We have successfully built our first machine learning model (k-NN classifier). Needless to say, our k-NN model was built on  a very small data set with very little optimization activities. So below are some ideas for improvemenst that did not make to our mini-project but can be implemented elsewhere in the future.

* One important note that we did not mentioned when we built our k-NN model is how do we **measure distances** between each data point, since there're more than just one way to measure distance in space. In our project, we use the default method to measure distance between two points, the **Euclidean distance**. The **Distances.jl** package provides user many other distance functions, so one can try to implement another method to see the changes in the overall performance. Further details can be found in the [documentation](https://github.com/JuliaStats/Distances.jl) the packages;

* We can further improve our model's performance by a series of **emsemble methods**. We will take a look at some of the related techiques in future projects.

#### 5.2 Final note

This section concludes our mini-project. We have built a model to predict the diabetic condition of a patients using their blood testing data. During the process, we walked through various common data science and data analysis tasks such as Exploratory Data Analysis, Data Visualisation, and so on. More challenging and sophisticated problems could be the idea of future project. 


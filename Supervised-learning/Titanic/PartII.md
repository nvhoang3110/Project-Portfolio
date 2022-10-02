# Titanic: Machine Learining from Disaster (Part II)

### Abstract

**The aim of the second part of the project**

This is the second part of our Titanic project, in which we will prepare our data before feeding it to a machine learning model. This involves various data preprocessing techniques such as feature engineering, feature extraction, missing data imputation and so on.

### Introduction

**Overview**
![image1](https://www.ncl.ac.uk/media/wwwnclacuk/research/images/boxes/researchimpact/Data_header2.jpg)

As we've explored the Titanic data set and gained a lot of insight  in the first part of our project, in the second part, we will preprocess our data before using it to train any machine learning model. 

> The formal definition is **Data Preprocessing** in machine learning is the process of transforming raw data into a useful, understandable format. 

Data preprocessing is no doubt, one of the most, if not the most, important step in any machine learning pipeline. Any algorithms that learn from the data and the learning outcome for problem solving heavily depends on the proper data needed to solve a particular problem. By cleaning and preprocessing the data, making it suitable for a particular machine learning model, we can greatly improve various model performance measures of our model later on, such as Accuracy, Cross-Entropy loss, compared to using raw, noisy data.

### Data Source

**Data set used in the second part of the project**

Just like in the first part, we will continued to use the ```training.csv``` data set. The path to the data can be found in the project repository. 

## Break down of the second part of our project

**Part II of our project is structured as follows:**

1. Installing, importing libraries and data used in our project;

2. Performing data preprocessing;

3. The last section is the final recap.

### 1. Install and import packages, data set

#### 1.1. Import the packages

We first start by loading all the libraries used in the second part of this project:

```Julia
# Load our packages to the machine
using CSV # To handle CSV file
using PrettyPrinting # For pretty printing
using DataFrames # Data frame in Julia
using MLJ # ML interface in Julia
```

#### 1.2. Load the data

For the data, we can load the ```.csv``` file to the environment and convert to standard ```DataFrame``` format:

```Julia
# Provide path to our data. Our data set can be found in project's repository
path = "training.csv"

# Load our data, convert from .csv format to DataFrame
titanic = DataFrame(CSV.File(path))
```

### 2. Data preprocessing

#### 2.1. Recapitulation: Some quick insights about our data

First, let's again take a glimpse at our data

```Julia
# Take a quick look at our data again
first(titanic, 10) |> pprint
```

```Julia
10×12 DataFrame
 Row │ PassengerId  Survived  Pclass  Name                               Sex      Age        SibSp  Parch  Ticket            Fare     Cabin      Embarked
     │ Int64        Int64     Int64   String                             String7  Float64?   Int64  Int64  String31          Float64  String15?  String1?
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │           1         0       3  Braund, Mr. Owen Harris            male          22.0      1      0  A/5 21171          7.25    missing    S
   2 │           2         1       1  Cumings, Mrs. John Bradley (Flor…  female        38.0      1      0  PC 17599          71.2833  C85        C
   3 │           3         1       3  Heikkinen, Miss. Laina             female        26.0      0      0  STON/O2. 3101282   7.925   missing    S
   4 │           4         1       1  Futrelle, Mrs. Jacques Heath (Li…  female        35.0      1      0  113803            53.1     C123       S
   5 │           5         0       3  Allen, Mr. William Henry           male          35.0      0      0  373450             8.05    missing    S
   6 │           6         0       3  Moran, Mr. James                   male     missing        0      0  330877             8.4583  missing    Q
   7 │           7         0       1  McCarthy, Mr. Timothy J            male          54.0      0      0  17463             51.8625  E46        S
   8 │           8         0       3  Palsson, Master. Gosta Leonard     male           2.0      3      1  349909            21.075   missing    S
   9 │           9         1       3  Johnson, Mrs. Oscar W (Elisabeth…  female        27.0      0      2  347742            11.1333  missing    S
  10 │          10         1       2  Nasser, Mrs. Nicholas (Adele Ach…  female        14.0      1      0  237736            30.0708  missing    C
  ```

Below are some of the most prominent problems with the data:

* Some features such as ```PassengerID``` or ```Ticket``` are just the identifiers for every passenger, and may not contain any useful information to predict whether the passenger was survived or not;

* ```SibSp``` and ```Parch``` columns represent information separated by siblings and spouses, and parents and children of passengers, respectively. So it may be more useful to combine these columns into a single variable containing overall family size;

* The ```Name``` variable itself is not very useful, however it does containt salutation information for every passenger, for example ```Mr```, ```Mrs``` or ```Miss```, which can be a very feature since it reflects the socioeconomic status of every passenger on board. We can extract this information and store it as a new variable;
* ```Age```, ```Cabin``` and ```Embarked``` are features with missing data, which requires proper treatment.

In the next section, we will address every single problem mentioned here, and propose solution for each each case.

#### 2.2. Feature engineering

We first create a new variable that captures the information of both ```SibSp``` and ```Parch``` columns, named it as ```FarmSize```. This new column indicates the number of family members each passenger travelled with:

```julia
# Create a new feature that captures information of both SiSp and Parch
titanic.FamSize = titanic.SibSp .+ titanic.Parch
```

```Julia
891-element Vector{Int64}:
 1
 1
 0
 1
 ⋮
 0
 3
 0
 0
```

#### 2.3. Feature extraction

We want to extract the salutation information from the ```Name``` column for every passengers. But then how many salutations are there in the data in total? This is when domain knowledge about our data set comes in handy. First, we will look for salutaions belonged to the following cases: ```Mr```, ```Mrs```, ```Miss```, ```Master```, ```Dr``` and ```Rev```. The rest we will put into the category ```Other```.

Let's look at one example from our ```Name``` variable to define a proper string pattern for our feature extraction task:

```Julia
# Take a look at one example
for i in 1:5
    println(titanic.Name[i])
end
```

```Julia
Braund, Mr. Owen Harris
Cumings, Mrs. John Bradley (Florence Briggs Thayer)
Heikkinen, Miss. Laina
Futrelle, Mrs. Jacques Heath (Lily May Peel)
Allen, Mr. William Henry
```

It looks like the salutation component is located in the middle of the name for each passenger, separated by a comma, followed by an empty space. The salutation starts with a capital letter of either ```M```, ```D``` or ```R```, follows by at least one letter and at most 5 letter and ends with a dot, separates with the rest of the name with another empty space. So the our pattern for string extraction should take the following format:

* Start with an empty space;

* Follow by one of the following capital letter: ```M```, ```D``` or ```R``` (the first letter of the salutation) and then at least one letter and at most five letters;

* End with a dot.

In Julia, we can define our pattern using **regular expression**:

```Julia
# Define string pattern
pattern = r"\s[MDR](\w+){1,5}\."
```

Next, we create an empty array to store our matching string, by iterating through all instances of the ```Name``` column and extracting parts that matches with our pattern:

```Julia
# Create an empty array to store our string
salutation = String[];

# Pattern matching, extracting and storing
for i in titanic.Name
    ex = match(pattern, i)
    if ex === nothing
        push!(salutation, "Other")
    else 
        push!(salutation, ex.match)
    end
end

# Inspect our array: the first 10 entries
println(first(salutation, 10))
```

```Julia
[" Mr.", " Mrs.", " Miss.", " Mrs.", " Mr.", " Mr.", " Mr.", " Master.", " Mrs.", " Mrs."]
```

Since our string still contains white space and dot, we can filter them out and then count the number of all salutation cases presented in the data:

```Julia
# Create an empty array to store our data
sal_clean = String[];

# Filter white space and dot. Push to the array above
for i in salutation
    sal = join(filter!(x -> !(x == '.' || x == ' '), [x for x in i]));
    push!(sal_clean, sal)
end;

# List of all salutations
words_list = unique(sal_clean)
```

```Julia
12-element Vector{Any}:
 "Mr"
 "Mrs"
 "Miss"
 "Master"
 "Don"
 "Rev"
 "Dr"
 "Mme"
 "Ms"
 "Major"
 "Other"
 "Mlle"
 ```

So we have more than just 7 initial guessed salutations. Let's count the number of entries for each of them:

```Julia
# Create an empty distionary to store our salutations and their number of entries
count_words = Dict{String, Int64}();

# Words counting
for i in sal_clean
    count_words[i] = get(count_words, i, 0) + 1
end

# Inspect the result
println(count_words)
```

```Julia
Dict("Don" => 1, "Master" => 40, "Miss" => 182, "Mrs" => 125, "Rev" => 6, "Other" => 7, "Major" => 2, "Mr" => 517, "Mme" => 1, "Mlle" => 2, "Ms" => 1, "Dr" => 7)
```

In total, we obtain 12 salutation categories, witht their number ranges from more than 500 entries to just 1. Here's a simple strategy to narrow down to just a couple of salutations: we will inspect any salutation that has less than 6 entries, as well as the ```Other``` category, and see if they contain any other useful information associated with other salutaions. If not, we would just convert them or just leave them as ```Other```. So now we consider ```Don```, ```Major```, ```Mme```, ```Mlle```, ```Ms``` and of course, ```Other```.

The entries of each categories can be accessed using index:

```Julia
# Get the index of our entries
other = findall(x -> x == "Other", sal_clean);

don = findall(x -> x == "Don", sal_clean);

Mme = findall(x -> x == "Mme", sal_clean);

Mlle = findall(x -> x == "Mlle", sal_clean);

Ms = findall(x -> x == "Ms", sal_clean);

Major = findall(x -> x == "Major", sal_clean);
```

Now let's inspect the true name in the data matched with the associated index. Starting with all the index of ```other``` category:

```Julia
# Inspect the true name in the data
for i in other
    println(titanic.Name[i])
end
```

```Julia
Duff Gordon, Lady. (Lucille Christiana Sutherland) ("Mrs Morgan")
Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")
Simonius-Blumer, Col. Oberst Alfons
Weir, Col. John
Crosby, Capt. Edward Gifford
Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)
Reuchlin, Jonkheer. John George
```

So it turns out that for the first two entries of ```Other```, salutations information exists, just in a different format (```Lady``` and ```Sir```). We can safely transform them into our predefined categories as suggested in the bracket (```Mrs``` and ```Mr```):

```Julia
# Transform salutation into predefined categories
sal_clean[other[1]] = "Mrs";

sal_clean[other[2]] = "Mr";
```

The rest of the entries we can safely leave them as it. With that being done, let's do the same thing for other categories:

```Julia
## -- Inspect the true name, convert salutation if necessary

# For "don" category
println(titanic.Name[don]) # No info can be retrieved. Transform into "Other"

sal_clean[don[1]] = "Other";
```

```Julia
# For "Mme" category
println(titanic.Name[Mme]) # No info can be retrieved. Transform into "Other"

sal_clean[Mme[1]] = "Other";
```

```Julia
# For "Mlle" category
for i in Mlle
    println(titanic.Name[i])
end # The second entry can be convert into "Mrs"

sal_clean[Mlle[1]] = "Other";

sal_clean[Mlle[2]] = "Mrs";
```

```Julia
# For "Ms" category
println(titanic.Name[Ms]) # Can convert to "Miss"

sal_clean[Ms[1]] = "Miss"
```

```Julia
# For "Major" category
for i in Major
    println(titanic.Name[i])
end # No info can be retrieved. Transform into "Other"

sal_clean[Major[1]] = "Other";

sal_clean[Major[2]] = "Other";
```

Everything is done. To make sure that all are correct, let's double check the new salutation categories again and count the number of entries for each of them:

```Julia
# Re-inspect our salutation variable
unique(sal_clean)
```

```Julia
7-element Vector{Any}:
 "Mr"
 "Mrs"
 "Miss"
 "Master"
 "Other"
 "Rev"
 "Dr"
 ```

```Julia
# Count the number of instance for each category
count_words = Dict{String, Int64}();

for i in sal_clean
    count_words[i] = get(count_words, i, 0) + 1
end

println(count_words)
```

```Julia
Dict("Miss" => 183, "Other" => 10, "Master" => 40, "Mr" => 518, "Mrs" => 127, "Rev" => 6, "Dr" => 7)
```

Feature extraction is done. We can finally create a new variable for the salutation, named as ```Salutation``` and then add to our data set as a new column:

```Julia
# Add new salutation column to our data
titanic.Salutation = sal_clean
```

```Julia
891-element Vector{Any}:
 "Mr"
 "Mrs"
 ⋮
 "Mr"
 "Mr"
 ```

#### 2.4. Feature selection

In every data set, some features are more useful for prediction than the others. In other words, they contain more predictive information and are more important for the modelling and predicting steps. So it's desirable omit variables with little no no prective value, to both reduce the computational cost as well as to improve the overall performance of the model by reducing noise. 

Let's look at the current variables of our feature space:

```Julia
# Inspect the number and names of features 
names(titanic)
```

```Julia
14-element Vector{String}:
 "PassengerId"
 "Survived"
 "Pclass"
 "Name"
 "Sex"
 "Age"
 "SibSp"
 "Parch"
 "Ticket"
 "Fare"
 "Cabin"
 "Embarked"
 "FamSize"
 "Salutation"
```

* From our data, knowing one ```PassengerID``` and ```Ticket``` is probably not going to help us to predict whether a passenger would survive or not, since these variables are unique for every passenger. So it's safe to drop them completely from the data;

* The ```Name``` column has salutation information, which we've already extracted and created as a new input feature. We can omit this feature as well;

* The ```Famsize``` column contains information of ```SibSp``` and ```Parch``` columns combine, so we drop both of them;

* We will handle the ```Cabin``` column in the next section, since it has ```missing``` data.

With that in mind, let's perform feature selection:

```Julia
# Perform feature selection
titanic = select!(titanic, Not([:PassengerId, :Ticket, :Name, :SibSp, :Parch]))
```

```Julia
891×9 DataFrame
 Row │ Survived  Pclass  Sex      Age        Fare     Cabin      Embarked  FamSize  Salutation 
     │ Int64     Int64   String7  Float64?   Float64  String15?  String1?  Int64    String
─────┼─────────────────────────────────────────────────────────────────────────────────────────
   1 │        0       3  male          22.0   7.25    missing    S               1  Mr
  ⋮  │    ⋮        ⋮        ⋮         ⋮         ⋮         ⋮         ⋮         ⋮         ⋮
                                                                               890 rows omitted
```

#### 2.5. Handling missing data

Recall from the first part of our project, we have missing data in three variables: ```Age```, ```Cabin``` and ```Embarked```. Let's investigate them further to see how many of them are missing entries, and theit proportion compared to the total number of entries:

```Julia
# Investigate the number of missing values for each column
des = describe(titanic, :nmissing)
```

```Julia
9×2 DataFrame
 Row │ variable    nmissing 
     │ Symbol      Int64    
─────┼──────────────────────
   1 │ Survived           0
   2 │ Pclass             0
   3 │ Sex                0
   4 │ Age              177
   5 │ Fare               0
   6 │ Cabin            687
   7 │ Embarked           2
   8 │ FamSize            0
   9 │ Salutation         0
```
We can add the ```percentage``` to see the portion of missing values over the total entries for each variables:

```Julia
# Add the percentage column
des.percentage = des.nmissing ./ nrow(titanic);

des
```

```Julia
9×3 DataFrame
 Row │ variable    nmissing  percentage 
     │ Symbol      Int64     Float64
─────┼──────────────────────────────────
   1 │ Survived           0  0.0
   2 │ Pclass             0  0.0
   3 │ Sex                0  0.0
   4 │ Age              177  0.198653
   5 │ Fare               0  0.0
   6 │ Cabin            687  0.771044
   7 │ Embarked           2  0.00224467
   8 │ FamSize            0  0.0
   9 │ Salutation         0  0.0
```

So from the table, it's straightforward to observe that:

* Since there are only 2 missing entries in the ```Embarked``` column, it's very easy to fill in or one can decide to just dump the entries out;

* The ```Age``` column contains 177 missing entries, which account for almost 20% of total data. We can find a way to infer them using missing data **imputation**;

* The ```Cabin``` column contains the most missing values, with 987 entries accounting for more than 77%. We could also find a way to fill the missing entries, but consider dropping this feature entirely is a better option since trying to induce that many missing entries can introduce bias to our model later.

With all that being said, our stategy is to first drop the ```Cabin``` column, and then use a built-in data transformation from MLJ, ```FillImputer```, to impute the missing data with a fixed value computed on the non-missing one:

```Julia
# Drop Cabin column
titanic = select!(titanic, Not(:Cabin))
```

Now in order to use the ```FillImputer```, just like when preparing data to feed a model in MLJ, we have to coerce all machine types into scientific type. Let's see what the current types of our data are:

```Julia
# Inspect the current machine types 
schema(titanic)
```

```Julia
┌────────────┬────────────────────────────┬─────────────────────────┐
│ names      │ scitypes                   │ types                   │
├────────────┼────────────────────────────┼─────────────────────────┤
│ Survived   │ Count                      │ Int64                   │
│ Pclass     │ Count                      │ Int64                   │
│ Sex        │ Textual                    │ String7                 │
│ Age        │ Union{Missing, Continuous} │ Union{Missing, Float64} │
│ Fare       │ Continuous                 │ Float64                 │
│ Embarked   │ Union{Missing, Textual}    │ Union{Missing, String1} │
│ FamSize    │ Count                      │ Int64                   │
│ Salutation │ Textual                    │ String                  │
└────────────┴────────────────────────────┴─────────────────────────┘
```

Right, so our input features are either ```Numeric``` (```Float``` and ```Int```) or ```String```, with of course ```Missing``` to form ```Union```. We can convert their types to proper scientific types as follows:

```Julia
# Convert feature's types into scientific types
titanic = coerce(titanic, :Survived => OrderedFactor, 
                          :Pclass => OrderedFactor, 
                          :Sex => Multiclass,
                          :Age => Continuous, 
                          :Fare => Continuous, 
                          :Embarked => Multiclass,
                          :FamSize => Count,
                          :Salutation => Multiclass);
```

And to make sure everything went well, we can double check the types of our data:

```Julia
# Inspect the scientific types of our data
schema(titanic)
```

```Julia
┌────────────┬───────────────────────────────┬───────────────────────────────────────────────────┐
│ names      │ scitypes                      │ types                                             │
├────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
│ Survived   │ OrderedFactor{2}              │ CategoricalValue{Int64, UInt32}                   │
│ Pclass     │ OrderedFactor{3}              │ CategoricalValue{Int64, UInt32}                   │
│ Sex        │ Multiclass{2}                 │ CategoricalValue{String7, UInt32}                 │
│ Age        │ Union{Missing, Continuous}    │ Union{Missing, Float64}                           │
│ Fare       │ Continuous                    │ Float64                                           │
│ Embarked   │ Union{Missing, Multiclass{3}} │ Union{Missing, CategoricalValue{String1, UInt32}} │
│ FamSize    │ Count                         │ Int64                                             │
│ Salutation │ Multiclass{7}                 │ CategoricalValue{String, UInt32}                  │
└────────────┴───────────────────────────────┴───────────────────────────────────────────────────┘
```

The data is ready for the imputation step. We load the imputer to the environment and connect it to our data:

```Julia
# Load the imputer
@load FillImputer
```

```Julia
[ Info: For silent loading, specify `verbosity=0`.
import MLJModels ✔
FillImputer
```

```Julia
# Connect our imputer with our data
filler = machine(FillImputer(), titanic)
```

```Julia
untrained Machine; caches model-specific representations of data
  model: FillImputer(features = Symbol[], …)
  args: 
    1:  Source @386 ⏎ Table{Union{AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{2}}, AbstractVector{OrderedFactor{3}}, AbstractVector{Union{Missing, Continuous}}, AbstractVector{Continuous}, AbstractVector{Union{Missing, Multiclass{3}}}, AbstractVector{Count}, AbstractVector{Multiclass{7}}}}
```

In the imputation step, missing data will be replaced by non-missing one. The default method is to use the ```median```. We train our imputer using all non-missing available data:

```Julia
# Perform data imputatiom
fit!(filler)
```

```Julia
[ Info: Training machine(FillImputer(features = Symbol[], …), …).
trained Machine; caches model-specific representations of data
  model: FillImputer(features = Symbol[], …)
  args:
    1:  Source @386 ⏎ Table{Union{AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{2}}, AbstractVector{OrderedFactor{3}}, AbstractVector{Union{Missing, Continuous}}, AbstractVector{Continuous}, AbstractVector{Union{Missing, Multiclass{3}}}, AbstractVector{Count}, AbstractVector{Multiclass{7}}}}
```

After training, the imputer now can be used to replace all missing value in the data:

```Julia
# Impute all missing value in the data
titanic = MLJ.transform(filler, titanic)
```

```Julia
891×8 DataFrame
 Row │ Survived  Pclass  Sex     Age      Fare     Embarked  FamSize  Salutation 
     │ Cat…      Cat…    Cat…    Float64  Float64  Cat…      Int64    Cat…
─────┼───────────────────────────────────────────────────────────────────────────
   1 │ 0         3       male       22.0   7.25    S               1  Mr
  ⋮  │    ⋮        ⋮       ⋮        ⋮        ⋮        ⋮         ⋮         ⋮
                                                                 890 rows omitted
```

We can then double-check our data to see whether it still contains any missing values or not:

```Julia
# Inspect our data to detect any missing values
describe(titanic, :nmissing)
```

```Julia
8×2 DataFrame
 Row │ variable    nmissing 
     │ Symbol      Int64    
─────┼──────────────────────
   1 │ Survived           0
   2 │ Pclass             0
   3 │ Sex                0
   4 │ Age                0
   5 │ Fare               0
   6 │ Embarked           0
   7 │ FamSize            0
   8 │ Salutation         0
```

After all of these steps, we finally obtain a clean and useful data set ready to be used to train any model. Let's save this data set for later use.

```Julia
# Save our clean data
CSV.write("titanic_clean.csv", titanic)
```

```Julia
"titanic_clean.csv"
```

And that concludes the data preprocessing step.

### 3. Conclusion

In the second part of the series, we have gone through various common data preprocessing steps like feature selection, feature engineering, missing data imputation, and so on. In the final part of the project, we will use this data to train a machine learning model and make prediction upon it.


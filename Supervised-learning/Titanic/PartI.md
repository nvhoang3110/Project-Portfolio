# Titanic: Machine Learining from Disaster (Part I)

### Abstract

**The aim of this project and the first part**

This project is an attemp to solve the famous and popular Machine Learning challenge, the Titanic shipwreck. The main goal is very straightforward: build a machine learning model to predict the survivability of passengers of this ship. In the part I of the project, we will explore the Titanic data set and analyse the data, as well as visualize it to gain more insight and understanding.

### Introduction

**Overview**

![RMS_Titanic](https://upload.wikimedia.org/wikipedia/commons/3/38/RMS_Titanic_4.jpg)

> RMS Titanic was a British passenger liner, operated by the White Star Line, which famously sank in the North Atlantic Ocean on 15 April, 1912 after stricking an iceberg during her maiden voyage from Southampton to New York city. Of the estimated 2.224 passengers and crew onboard, more than 1.500 died, making it the deadliest sinking of a single trip at that time.The disaster drew public attention, provided foundational material for the disaster film genre, and has inspired many artistic works. 

The Titanic shipwreck is a very popular machine learning challenge among the machine learning community, with the main goal is to build a classifier in order to predict the survivability of passengers and crew onboard. This project is another attempt to solve the problem, using the Julia programming language and MLJ ecosystem.

The project is divided into 3 parts:

* In part I, we will investigate the data set in-depth, and perform various common data analysis tasks and data visualization;

* In part II, we tackle data preprocessing step;

* In part III, we build a machine learning model and evaluate its performance.

### Data Source

**Data used in the part I of the project**

The data of passengers and crew onboard is publicly available for everyone to access. There are various sources on the internet that one can use and download, but we are going to use the data set comes from [Kaggle](https://www.kaggle.com/), an online community of data scientists and machine learning practioners. Here's a quick [link](https://www.kaggle.com/competitions/titanic/data) get access to out data. Data are in ```.csv``` file format, and has been split into two groups:

* Training set (```train.csv```);

* Test set (```test.csv```).

For the first part and also the second part of our project, we will use the training data set only. 

### Break down of the first part of our project

**Part I of our project is structured as follows:**

1. Installing, importing packages and data used in our project;

2. Performing EDA and Data Visualization;

3. The last section is the final recap.

### 1. Install and import packages, data set

#### 1.1. Install, import the packages

We first start by installing all the packages used in your project and loading all the libraries used in the first part:

```Julia
# Define all packages that we are going to use through out this project
packages = ["CSV", "PrettyPrinting", "DataFrames", "Pipe", "CairoMakie", "MLJ", "StableRNGs"]
```

```Julia
# Install our packages 
import Pkg; Pkg.add(packages)

# Load our packages to the machine
using CSV # To handle CSV file
using PrettyPrinting # For pretty printing
using DataFrames# Data frame in Julia
using Pipe # For piping 
using CairoMakie # For plotting
```

#### 1.2. Load the data

For the data, we can load the ```.csv``` file to the environment and convert to standard ```DataFrame``` format:

```Julia
# Provide path to our data. Our data set can be found in project's repository
path = "training.csv"

# Load our data, convert from .csv format to DataFrame
titanic = DataFrame(CSV.File(path))
```

### 2. Exploratory Data Analysis (EDA) and Data Visualization

#### 2.1. Data understanding

First, let's take a quick look at our data:

```Julia
# Take a quick look at our data
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

We have a ```DataFrame``` that contains in total 891 observations (instances) over 12 variables, including 11 feature variables (```PassengerID```, ```Pclass```, ```Name```, ```Sex```, ```Age```, ```SibSp```, ```Parch```, ```Ticket```, ```Fare```, ```Cabin``` and ```Embarked```) and one target variable (```Survived```). This idea is confirmed by inspecting the dimensions of our data:

```Julia
# Inspect dimension of our data
size(titanic)
```

```Julia
(891, 12)
```

Below are detailed descriptions of all variables in the data set:

* ```PassengerID``` - An arbitrary number unique to each passenger on board;

* ```Survived``` - An integer denoting survival (1 = survived, 0 = died);

* ```Pclass``` - A proxy for socio-economic status (SES). Including 1-st, 2-nd and 3-rd class;

* ```Name``` - A character vector of the passengers’ names;

* ```Sex``` - A character vector containing gender of passengers. Including “male” and “female”;

* ```Age``` - The age of passenger;

* ```SibSp``` - The combined number of siblings and sprouses on board;

* ```Parch``` - The combined number of parents and childrens on board;

* ```Ticket``` - A character vector with each passenger’s ticket number;

* ```Fare``` - The amount of money each passenger paid for their ticket;

* ```Cabin``` - A character vector of each passenger’s cabin number;

* ```Embarked``` - A character vector of which port passengers embarked from;

So to sum up everything so far, the data set contains in total 8 categrical and 4 continuous variables, stretching over 891 entries. Our goal is to predict the ```Survived``` column, using other information from the data set.

To get more insights about the distribution of our data, we can take a look at the statistic summary:

```Julia
# Statistic summary
describe(titanic, :all)
```

```Julia
12×13 DataFrame
 Row │ variable     mean      std       min                  q25     median   q75     max                          nunique  nmissing  first                    last                 ⋯
     │ Symbol       Union…    Union…    Any                  Union…  Union…   Union…  Any                          Union…   Int64     Any                      Any                  ⋯
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ PassengerId  446.0     257.354   1                    223.5   446.0    668.5   891                                          0  1                        891                  ⋯
   2 │ Survived     0.383838  0.486592  0                    0.0     0.0      1.0     1                                            0  0                        0
   3 │ Pclass       2.30864   0.836071  1                    2.0     3.0      3.0     3                                            0  3                        3
   4 │ Name                             Abbing, Mr. Anthony                           van Melkebeke, Mr. Philemon  891             0  Braund, Mr. Owen Harris  Dooley, Mr. Patrick
   5 │ Sex                              female                                        male                         2               0  male                     male                 ⋯
   6 │ Age          29.6991   14.5265   0.42                 20.125  28.0     38.0    80.0                                       177  22.0                     32.0
   7 │ SibSp        0.523008  1.10274   0                    0.0     0.0      1.0     8                                            0  1                        0
   8 │ Parch        0.381594  0.806057  0                    0.0     0.0      0.0     6                                            0  0                        0
   9 │ Ticket                           110152                                        WE/P 5735                    681             0  A/5 21171                370376               ⋯
  10 │ Fare         32.2042   49.6934   0.0                  7.9104  14.4542  31.0    512.329                                      0  7.25                     7.75
  11 │ Cabin                            A10                                           T                            147           687  missing                  missing
  12 │ Embarked                         C                                             S                            3               2  S                        Q
                                                                                                                                                                     1 column omitted
```

Here are some impressions:

* Only **38%** of passengers survived the horrific maritime incident. This was in fact, the deadliest sinking of a single ship up to that time;

* The mean values of both ```SibSp``` and ```Parch``` column are relatively small, with standard deviation varies in a very minor degree. This highly suggests that most passengers travelled alone, with no family members with them. The highest combined number of siblings and parents/children on board are 8 and 6, respectively;

* The majority of passengers belongs to the second and third-class. On average, they had to pay **32£** for their ticket;

* The ```Age``` of passengers range from just 0.4 (newborns) to 80 years old. The average passenger was 30 years old at that time;

* There are missing values in some of our features, like 177 missing entries in the ```Age``` variable. We will take a look at them closer later.

To have a better visual understanding of how our data look like and how they interact with each other, it's a good idea to plot them. 

#### 2.2. Data visualization

> Unfortunately, since ```Makie.jl``` was built to optimize low-level graphical objects, to make statistical plots using the package, we have to set up everything from data transformation from wide to long format, color setting, and so on.  

Using plot is a great way to understand our data, and that's what are are going to do in this section. We can have a pretty good picture of how the passengers on board that day were like by plotting their genders and classes, and where do they came from as well. We first start with gender first: How many of them were men and how many of them were women. We also add their accociated classes (```Pclass```) to the plot:

```Julia
# Gender and class bar plot
gender_and_pclass = begin
    
    # Create a helper function
    function sex_and_pclass(x::String, y::Int64)
        n = nrow(subset(titanic, :Sex => ByRow(==(x)), 
                                :Pclass => ByRow(==(y))))
        return n
    end

    # Filter data for ploting
    plot_dat = DataFrame(
                Female = [sex_and_pclass("female", 1), sex_and_pclass("female", 2), sex_and_pclass("female", 3)],
                Male = [sex_and_pclass("male", 1), sex_and_pclass("male", 2), sex_and_pclass("male", 3)]);

    # Transform data into vector for plotting
    plot_vec = @pipe plot_dat |> Matrix |> transpose |> reshape(_, (6, 1)) |> vec

    # Reset all modification to default
    set_theme!();

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080));

    # Update theme and font size
    update_theme!(with_theme = theme_light(), fontsize = 30);

    # Initialize axis
    axis = fig[1,1] = Axis(fig, title = "",
                                xlabel = "Pclass",
                                xlabelfont = "TeX Gyre Heros Makie Bold",
                                ylabel = "Count",
                                ylabelfont = "TeX Gyre Heros Makie Bold")

    # Plot
    barplot!(axis, repeat(1:3, inner = 2), plot_vec, 
            dodge = repeat(1:2, outer = 3), 
            color = map(x -> x == 1 ? "#c94c4c" : "#034f84", repeat(1:2, outer = 3)));

    # Add legend
    leg = fig[1, 2] = Legend(fig, 
                            [[PolyElement(color = "#c94c4c")], [PolyElement(color = "#034f84")]], 
                            ["Female", "Male"], 
                            "Sex",
                            halign = :left,
                            valign = :center,
                            tellwidth = true, 
                            tellheight = false,
                            framevisible = false);

    # Return the final plot
    fig
end
```

![sex_and_pclass](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Titanic/Plots/sex_and_pclass.png?raw=true)

From the plot, it's clear that the majority of passengers were of the third-class. Also, there were more men than women in every class, but for the third-class the difference was significant: the number of men were as twice the number of women. And there were a little bit more people belonged to the first class than to the second-class, though the numbers for both classes are quite similar.

The next thing we are going to visualize is where did the passengers come from. In other words, at which port did they depart. Just like last plot, we add another variable to the plot to better separate the classes. In this plot we can use passenger's classes:

```Julia
# Embarked and Pclass plot
embarked_and_pclass = begin
    
    # Define helpter function for filtering and coloring
    function embarked_and_pclass(x, y)
        new_dat = select(titanic, [:Embarked, :Pclass]) |> dropmissing
    
        n = nrow(subset(new_dat, :Embarked => ByRow(==(x)), 
                                :Pclass => ByRow(==(y))))
        return n
    end

    function col_con(x)
        if x == 1
            return "#c94c4c"
        elseif x == 2
            return "#034f84"
        else
            return "#deeaee"
        end
    end

    # Filter data for ploting
    plot_dat = DataFrame(
                C = [embarked_and_pclass("C", 1), 
                    embarked_and_pclass("C", 2), 
                    embarked_and_pclass("C", 3)],
                Q = [embarked_and_pclass("Q", 1), 
                    embarked_and_pclass("Q", 2), 
                    embarked_and_pclass("Q", 3)],
                S = [embarked_and_pclass("S", 1), 
                    embarked_and_pclass("S", 2), 
                    embarked_and_pclass("S", 3)]);
    
    # Transform data into vector for plotting
    plot_vec = @pipe plot_dat |> Matrix |> reshape(_, (9, 1)) |> vec

    # Reset all modification to default
    set_theme!();

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080));

    # Update theme and font size
    update_theme!(with_theme = theme_light(), fontsize = 30);

    # Initialize axis
    axis = fig[1,1] = Axis(fig, title = "",
                            xlabel = "Embarked",
                            xticks = (1:3, ["C", "S", "Q"]),
                            xlabelfont = "TeX Gyre Heros Makie Bold",
                            ylabel = "Count",
                            ylabelfont = "TeX Gyre Heros Makie Bold");
    
    # Plot
    barplot!(axis, repeat(1:3, inner = 3), plot_vec, 
            dodge = repeat(1:3, outer = 3), 
            color = map(col_con, repeat(1:3, outer = 3)));

    # Add legend
    leg = fig[1, 2] = Legend(fig, 
                            [[PolyElement(color = "#c94c4c")], 
                            [PolyElement(color = "#034f84")],
                            [PolyElement(color = "#deeaee")]], 
                            ["1", "2", "3"], 
                            "Pclass",
                            halign = :left,
                            valign = :center,
                            tellwidth = true, 
                            tellheight = false,
                            framevisible = false);
    
    # Return the final plot
    fig
end
```

![embarked_and_class](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Titanic/Plots/embarked_and_pclass.png?raw=true)

Quite interesting. So it looks like the majority of first-class passengers were departed from port ```Q``` and also port ```C```, and so did the third-class passengers for port ```Q```. There is a high chance that port ```Q``` was  the biggest or the main port, since the cruiser received most of its passengers here. For port ```S```, there were verly little to no passengers from other classes departed from this port other than third-class passengers. Port ```C``` has a very different patent from other ports: first-class passengers from this port outnumbered either second-class passengers or third-class passengers. 

So after inspecting the passengers more closely based on the available data, let's move our attention to the only target variable of the data set: ```Survived```. As we said earlier, the survival rate is less than 40%, so let's visualize the total number of survived and not-survived passengers first:

```Julia
survivability_plot = begin

    # Filter data based on survivability
    survived = filter(:Survived => x -> x == 1, titanic);

    not_survived = filter(:Survived => x -> x == 0, titanic);

    # Initialize helper dictionaries
    survivability = Dict(survived => 1, not_survived => 2);

    color_con = Dict(survived => "#5b9aa0", not_survived => "#622569");

    # Select plot font
    plot_font = "TeX Gyre Heros Makie Bold";

    # Reset all modification to default
    set_theme!();

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080));

    # Update theme and font size
    update_theme!(with_theme = theme_light(), fontsize = 30);

    # Initialize axis
    axis = fig[1, 1] = Axis(fig, title = "", 
                            xlabel = "",
                            xticks = (1:2, ["Survived", "Not survived"]),
                            xlabelfont = plot_font,
                            ylabel = "Count",
                            ylabelfont = plot_font);

    # Plot
    for i in [survived, not_survived]
        barplot!(axis, survivability[i], nrow(i), color = color_con[i])
    end

    # Return the final figure
    fig
end
```

![barplot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Titanic/Plots/bar_plot.png?raw=true)

This plot is self-explanatory. As we have mentioned earlier, around 40% of passengers survived the shipwreck. So there is a skewed in our data toward the ```not_survived``` side. One problem with this plot is that it does not give us much information about how other **Social & Economic** factors affect the survivability of passengers. To gain such insight, for the categorical variables in the data, we can plot the percentage of survived passengers and not-survived passengers belonged to each class, and then stack them up to compare. We'll that for ```Pclass```, ```Sex``` and also for ```Embarked``` variables. Note than since column ```Embarked``` has missing values, we have to drop them before plotting, otherwise it would result in an error:

```Julia
## -- Stacked percentage barplot

# First factor: Pclass
first_bar = begin

    # Define helper function
    function my_fil(dat, col, i)
        return nrow(filter(col => x -> x == i, dat))
    end

    # y-axis format
    plot_range = (0:25:100, ["0%", "25%", "50%", "75%", "100%"])

    # Reset all modification to default
    set_theme!()

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080))

    # Update theme and font size
    update_theme!(with_theme = theme_light(), fontsize = 30)

    # Initialize the axis
    axis1 = fig[1, 1] = Axis(fig, title = "Pclass", 
                        xlabel = "",
                        xticks = (1:3, ["1", "2", "3"]),
                        xlabelfont = plot_font,
                        ylabel = "Percent",
                        yticks = plot_range,
                        ylabelfont = plot_font)

    for i in 1:3
        n = my_fil(survived, :Pclass, i) / my_fil(titanic, :Pclass, i) * 100
        barplot!(i, 100, color = "#622569")
        barplot!(i, n, color = "#5b9aa0")
    end
end
```

```Julia
# Second factor: Sex
second_bar = begin

    # Initialize helper dictionary
    gender = Dict("female" => 1, "male" => 2)

    # Initialize the axis
    axis1 = fig[1, 2] = Axis(fig, title = "Sex", 
                        xlabel = "",
                        xticks = (1:2, ["Female", "Male"]),
                        xlabelfont = plot_font,
                        ylabel = "Percent",
                        yticks = plot_range,
                        ylabelfont = plot_font)

    for i in ["female", "male"]
        n = my_fil(survived, :Sex, i) / my_fil(titanic, :Sex, i) * 100
        barplot!(gender[i], 100, color = "#622569")
        barplot!(gender[i], n, color = "#5b9aa0")
    end
end
```

```Julia
# Third bar and add legend
third_bar = begin

    # Initialize helper dictionary
    embarked = Dict("S" => 1, "C" => 2, "Q" => 3)

    # Initialize the axis
    axis3 = fig[2, 1] = Axis(fig, title = "Embarked", 
                        xlabel = "",
                        xticks = (1:3, ["S", "C", "Q"]),
                        xlabelfont = plot_font,
                        ylabel = "Percent",
                        yticks = plot_range,
                        ylabelfont = plot_font)

    # Plot
    for i in ["S", "C", "Q"]
        n = my_fil(survived |> dropmissing, :Embarked, i) / my_fil(titanic |> dropmissing, :Embarked, i) * 100
        barplot!(embarked[i], 100, color = "#622569")
        barplot!(embarked[i], n, color = "#5b9aa0")
    end

    # Add legend 
    leg = fig[2, 2] = Legend(fig, 
                        [[PolyElement(color = "#5b9aa0")], [PolyElement(color = "#622569")]], 
                        ["Survived", "Not survived"], 
                        "Status",
                        rowgap = 3,
                        halign = :center,
                        valign = :center,
                        tellwidth = false, 
                        tellheight = false,
                        framevisible = false)

    # Return the final plot
    fig
end
```

![multiplebar_plot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Titanic/Plots/multibar_plot.png?raw=true)

Here are some insight drew from our plots:

* Passengers belonged to the first-class had a higher change to survive than second-class passengers and third-class passenger. In fact, while the second-class passengers had around 50% chance to survive, third-class passengers had less than 25% chance;

* Nearly three-fourths of female passengers survived, while the figure for male is less than one-fourths. So if a passenger was a man, he would have less than 25% chance to survive, while the other way around was true if the passenger was a woman;

* Port that passengers departed from does not seem to have a strong effect on the survivability rate in general. However, passengers departed from port ```Q``` had the lowest survivability rate, with only around 50% chance, which makes sense since from the previous plot, we comfirmed that port ```Q``` was the biggest port (there were a lot of passengers departed from this port).

How about other continuous variables, such as ```Age```, ```Fare``` and ```Parch``` ? To observe their distribution and at the same time to compare the diffecences in distributions between each group (in this case ```survived``` and ```not-survived```), let's use the **violin plot** to do so:

```Julia
# Violin plot
violin_plot = begin

    # Define plot range and annotation
    plot_range = (0:1, ["Not survived", "Survived"])

    # Initialize empty scene 
    fig = Figure(resolution = (1920, 1080))

    # Update theme and font size
    update_theme!(fontsize = 30)

    # Initialize axis
    axis = fig[1, 1] = Axis(fig, title = "Age", 
                            ylabel = "Count",
                            ylabelfont = plot_font,
                            xticks = plot_range)

    axis2 = fig[1, 2] = Axis(fig, title = "Fare", xticks = plot_range, yticks = (0:300:600));

    axis3 = fig[1, 3] = Axis(fig, title = "SibSp", xticks = plot_range);

    axis4 = fig[1, 4] = Axis(fig, title = "Parch", xticks = plot_range);

    # Drop the missing in the data
    titanic_nomissing = titanic |> dropmissing

    # Plot
    violin!(axis, titanic_nomissing.Survived, titanic_nomissing.Age, color = "#92a8d1")

    violin!(axis2, titanic_nomissing.Survived, titanic_nomissing.Fare, color = "#b1cbbb");

    violin!(axis3, titanic_nomissing.Survived, titanic_nomissing.SibSp, color = "#deeaee");

    violin!(axis4, titanic_nomissing.Survived, titanic_nomissing.Parch, color = "#034f84");

    # Return the final plot
    fig
end
```

![violin_plot](https://github.com/nvhoang3110/Project-Portfolio/blob/main/Supervised-learning/Titanic/Plots/violin_plot.png?raw=true)

From the plot, we can have some observations:

* It seems like passengers who survived tended to have slightly more family members on board, although passengers with very large families on board tended not to survive;

* ```Age``` doesn’t seem to have had an obvious impact on survival, however, if passengers were below their 50s, they have a better chance to survive;

* Paying more for the ticket definitely increases passenger survivability.

### 3. Final recap

In the first part of the project, we did some extensive data analysis and data visualization for the titanic data set, and gained a lot more insight and perspective tfrom it. In the next part of the series, we will process our data set, a very important step in any machine learning pipeline.


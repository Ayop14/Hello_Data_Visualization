# HELLO DATA VISUALIZATION 

## Motivation
This is my first project in my data science specialization roadmap. The idea is to design my own projects and solve real world projects that could arise during **real scenarios**. 

As a Junior computer scientist I realized I have a long way to learn about the field. That' s why I hand designed a roadmap for myself, inspired in MITT, Harvard and popular Data science courses. This is the first stone towards being a profesional ML engineer. 

## Introduction

This project focuses on the **science** behind data visualization, not any data visualization software in particular. The idea is to know *why* certain visualization work and which doesn't. Psicology, intuition, **and making a point** have a huge role in it.

Its focused on the book **Fundamentals of Data Visualization**, by **Claus O. Wilke**, and supported on a few youtube videos, tutorials, and *junior computer scientist stack overflow searching shenanigans*.

## Objective

The objective is to make practice about the points the book makes. Try to create visualizations where knowing that chapter information truly makes a diference, or solves a problem. 

To make visualizations I need ***DATA***! I've been really interested in 125CC bikes recently, so I build a dataset with lots of information about the current offer of 125CC bikes. That will motivate myself to ask the correct, and if possible complicated questions about the data. At the same time, it might clear my mind on which one to choose, hehe~

The idea is to read a chapter, and make a question to the dataset that can be answered with a correctly designed data visualization. 

## Dataset
As previously stated, it's a dataset of new 125CC Bikes. In the future, I don't discard making the same for 125-500CC bikes. The features are as follows:

- **Brand**: Brand that offers the Bike
- **Name**: Name of the bike model
- **Consumption (km/l)**: Gasoline litters consumed after driving 100 Km. Obtained from https://www.moto125.cc/ . Only null in case the web didnt had data from the bike. The consumption is worst-case possible. 
- **Acceleration (0-100km/h)**: Time (s) it takes the bike to reach 100 km/h. Obtained from https://www.moto125.cc/ Empty if the bike doesnt reach that speed or if the web didnt provide a value. 
- **Max Speed (Km/h)**:Max speed of the bike. Obtained from https://www.moto125.cc/ Empty if the web didnt had data from the bike. Max speed obtained in plain terrain, no inclination. 
- **Bike type**: Type of bike depending on its overall characteristics. Options are: Naked, Sport, Trail, Scrambler, Custom or Scooter. 
- **Power (kw)**: Power the bike's engine has. EU limits small bikes to 11KW
- **Rpm max power**: RPMs the engine provides its max power output from. 
- **Cilinders**: Number of cylinders the bike's engine has
- **Refrigeration**: Type of refrigeration the engine has. Options are liquid or air. 
- **Torque (Nm)**: Torque the bike has. 
- **Rpm max torque**: RPMs the engine provides its max torque power
- **Weight full (kg)**: Weight of the bike with a full tank
- **Seat height (mm)**: Distance from the seat to the ground in milimeters
- **Gasoline capacity (l)**: Gasoline Capacity for the bike's tank 
- **Price (eur)**: Price of the bike in euros
- **Number of gears**: Number of gears the bike has
- **Brake pistons**: Number of brake pistons in the front brake
- **Front brake disk size (mm)**: Front brake's disk size
- **Back brake disk size (mm)**: Back brake's disk size
- **Length (mm)**: Length of the bike
- **Width (mm)**: Length of the bike
- **Height (mm)**: Length of the bike

## Legal details...?
I do not own the right of any of these models, and the information in the dataset might not be accurate. Some data had to be assumed, like "weight full". I added the tank capacity in l to the empty weight as an estimation in some cases.

## Future Work

This is only the first step in my roadmap. Regarding data visualization, I also want to make an interactive, real-time dashboard using dash or some similar technology. My current dataset doesnt allow me to show real-time data, so I will take that on in another project from the roadmap. One that allows me to build a useful tool, rather than a dummy one like this dataset.

Next step for me will be a popular tool: PANDAS! I've been using it for quite a few months now, but I can only begin to imagine what a powerful tool can be if someone dominates it.

## Final Words

Project in development - Once I finish
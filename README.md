# Hackthon Ignite camp

This was a project elaborated on november 29th, 30th and december 1st of 2019 for a hackathon organizated by Scotiabank | Colpatria. They planted two problems to solve, the first one were about the cash distribution among their cash machines and the other one were about the manage of the vacation period of their employees.

My team and me selected the second problem. There are some persons that they don't like rest, in other words,are addicted to work. This is a problematic related to mental deseases.

Our solution proposal was a system based on AI that suggests the vacational time for the employees, depending on variables like: Last rest period, labor impediments, if is woman and was recently pregnant and similar facts. The core of the project is the AI that works with an ARIMA (autoregressive integrated moving average) model, which indicates a percetage of probability to the rest time for the employees. If the probablity is more than 100%, indicates thet the person have much time of vacation accumulated.

## The interface

Formally the project were named Xamay, which means "rest" in Quechua language. The interaface of this project was made with Bootstrap, and the display of some dashboards was made with PowerBI.

### Overview
![mainVIewXamay](https://user-images.githubusercontent.com/43974127/110966553-5b130800-8323-11eb-8182-fc35d6096671.png)

As you can see, the data of the main suggestions is showed in a table, the critical data is displayed in red to indicates to the manager the urgency of each case.

## License

The current state of the project is under [MIT license](https://opensource.org/licenses/MIT).

## Final note

With this project with my team won the hackathon.

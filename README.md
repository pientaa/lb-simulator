# lb-simulator

## Table of contents

- [lb-simulator](#lb-simulator)
  * [Table of contents](#table-of-contents)
  * [Generator](#generator)
      - [Run](#run)
      - [Tasks distribution](#tasks-distribution)
    + [Example of generated data](#example-of-generated-data)
      - [Requests](#requests)
      - [Load vectors](#load-vectors)
  * [Simulator](#simulator)
 
## Generator

Generator takes following parameters:
- `num_of_shards`, which defines number of vectors that will be generated - it's identical to number of shards located in total on cloud.
- `num_of_samples`, which defines nomber of periods that will be considered when generating tasks.
- `period`, which defines the amount of time related to one element of load vector.
- `shape`, which defines &alpha; parameter of gamma distribution from which number of tasks per period will be drawn.
- `size`, which defines &beta; parameter of gamma distribution from which number of tasks per period will be drawn.
 
#### Run

Generate requests and load vectors using following function from `generator` module:
```python
def generator(num_of_shards, num_of_samples, new_period, shape, scale)
```
#### Tasks distribution

Tasks are drawn from gamma distribution, but to be precise - as far as `shape` parameter is scalar -  it's Erlang distribution, whose mean and standard deviation can be calulated with following equations:

 &mu; = &alpha; * &beta; </br>
 &sigma; =  &beta; * √&alpha;
 
### Example of generated data

```python
generator(5 10 5 2 2)
```


#### Requests

|id|timestamp|shard|load |
|------|---------|-----|-----|
|0     |0.575    |4    |3.587|
|1     |0.595    |2    |2.481|
|2     |4.93     |3    |2.566|
|3     |5.08     |3    |4.547|
|4     |5.925    |1    |15.63|
|5     |9.915    |3    |3.536|
|6     |10.19    |2    |2.098|
|7     |10.43    |5    |1.783|
|8     |10.545   |4    |0.606|
|9     |10.645   |4    |2.185|
|10    |11.095   |2    |3.723|
|11    |11.74    |3    |5.347|
|12    |11.84    |4    |4.333|
|13    |12.53    |4    |13.207|
|14    |13.175   |4    |8.572|
|15    |15.505   |3    |1.407|
|16    |16.205   |1    |7.395|
|17    |19.825   |5    |1.8  |
|18    |20.62    |4    |2.242|
|19    |22.205   |3    |4.986|
|20    |22.5     |2    |1.005|
|21    |23.675   |2    |0.899|
|22    |25.98    |4    |2.884|
|23    |28.4     |4    |8.25 |
|24    |28.535   |5    |1.857|
|25    |30.925   |1    |8.524|
|26    |31.135   |2    |1.405|
|27    |31.795   |3    |3.347|
|28    |34.43    |3    |0.763|
|29    |38.32    |1    |2.269|
|30    |38.74    |1    |2.793|
|31    |40.005   |5    |1.767|
|32    |40.105   |5    |8.903|
|33    |40.23    |2    |4.558|
|34    |40.235   |4    |2.15 |
|35    |40.295   |3    |12.854|
|36    |40.365   |3    |5.065|
|37    |40.365   |1    |0.078|
|38    |40.71    |1    |1.705|
|39    |40.785   |2    |3.965|
|40    |41.13    |5    |6.327|
|41    |41.285   |5    |1.459|
|42    |42.02    |5    |7.739|
|43    |44.05    |5    |1.866|
|44    |45.01    |2    |1.985|
|45    |45.05    |3    |0.505|
|46    |45.16    |3    |9.694|
|47    |45.765   |5    |0.628|
|48    |45.89    |4    |8.159|
|49    |46.435   |1    |2.392|
|50    |46.705   |4    |2.158|
|51    |46.895   |3    |8.148|
|52    |47.25    |3    |4.49 |
|53    |48.16    |1    |3.889|



#### Load vectors

| [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] | [9] | [10] | [11] | [12] 
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|---|
|0.0  |1.815|1.0  |0.759|3.283|0.0  |0.815|1.478|0.782|0.846|0.41 |0.0  |
|0.496|0.0  |1.165|0.0  |0.381|0.0  |0.281|0.0  |1.705|0.397|0.0  |0.0  |
|0.014|1.425|1.342|0.698|0.559|0.438|0.755|0.067|2.868|3.326|3.502|1.305|
|0.717|0.0  |4.049|1.235|1.889|4.04 |0.0  |1.386|0.43 |1.254|0.81 |0.0  |
|0.0  |0.0  |0.357|0.035|0.325|0.293|0.078|0.0  |3.184|2.554|0.0  |0.0  |

## Simulator

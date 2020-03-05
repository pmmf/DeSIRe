# DeSIRe
Deep Signer-Invariant Representations for Sign Language Recognition

## Description
Source code for the implementation of **DeSIRe**, a novel deep neural network that aims to learn **De**ep **S**igner-**I**nvariant **Re**presentations, as described in the paper:

P. M. Ferreira, D. Pernes, A. Rebelo and J. S. Cardoso, "**[DeSIRe: Deep Signer-Invariant Representations for Sign Language Recognition](https://ieeexplore.ieee.org/abstract/document/8937777)**", *in IEEE Transactions on Systems, Man, and Cybernetics: Systems*. doi: 10.1109/TSMC.2019.2957347

| [![page1](./imgs/DeSIRe.png)](https://ieeexplore.ieee.org/abstract/document/8937777)  |
|:---:|

## Example of Usage

***Train:*** 
~~~bash
python code/run_twins.py --model=twins --dataset=staticSL --gpu=0 --mode=train
~~~

***Test:*** 
~~~bash
python code/run_twins.py --model=twins --dataset=staticSL --gpu=0 --mode=test
~~~

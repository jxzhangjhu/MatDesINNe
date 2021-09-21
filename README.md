# MatDesINNe



## Brief Introduction
The ability to readily design novel materials with chosen functional properties on-demand represents a next frontier in materials discovery. However, thoroughly and efficiently sampling the entire design space in a computationally tractable manner remains a highly challenging task. To tackle this problem, we propose an inverse design framework (**MatDesINNe**) utilizing invertible neural networks which can map both forward and reverse processes between the design space and target property. This approach can be used to generate materials candidates for a designated property, thereby satisfying the highly sought-after goal of inverse design. We then apply this framework to the task of band gap engineering in two-dimensional materials, starting with MoS2. Within the design space encompassing six degrees of freedom in applied tensile, compressive and shear strain plus an external electric field, we show the framework can generate novel, high fidelity, and diverse candidates with near-chemical accuracy. We extend this generative capability further to provide insights regarding metal-insulator transition, important for memristive neuromorphic applications among others, in MoS2 which is not otherwise possible with brute force screening. This approach is general and can be directly extended to other materials and their corresponding design spaces and target properties.

## Getting Started

You will need [Python 3.6](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

## Datasets
The datasets used in DevNet are also released here. See our anomaly detection dataset repository [ADRepository](https://github.com/GuansongPang/anomaly-detection-datasets) for more preprocessed datasets that are widely-used in other papers.


## Credits

Some code of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.

## Citation
```bibtex
@article{fung2021inverse,
  title={Inverse design of two-dimensional materials with invertible neural networks},
  author={Fung, Victor and Zhang, Jiaxin and Hu, Guoxiang and Ganesh, P and Sumpter, Bobby G},
  journal={arXiv preprint arXiv:2106.03013},
  year={2021}
}
```

## License

This project is licensed under the MIT License.

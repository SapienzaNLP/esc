# ESC: Redesigning WSD with Extractive Sense Comprehension
In ESC ([Barba et al., 2021](https://www.aclweb.org/anthology/2021.naacl-main.371/)) we redesigned Word Sense 
Disambiguation ([Navigli et al., 2009](http://wwwusers.di.uniroma1.it/~navigli/pubs/ACM_Survey_2009_Navigli.pdf)) as an Extractive Reading Comprehension task and achieved unprecedented performances on a number of 
different benchmarks and settings. In this repo we provide the code to reproduce the results of the paper along with the
checkpoints for the best models.

## How to Cite
```
@inproceedings{barba-etal-2021-esc,
    title = "{ESC}: Redesigning {WSD} with {E}xtractive {S}ense {C}omprehension",
    author = "Barba, Edoardo  and
      Pasini, Tommaso  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.371",
    pages = "4661--4672",
    abstract = "Word Sense Disambiguation (WSD) is a historical NLP task aimed at linking words in contexts to discrete sense inventories and it is usually cast as a multi-label classification task. Recently, several neural approaches have employed sense definitions to better represent word meanings. Yet, these approaches do not observe the input sentence and the sense definition candidates all at once, thus potentially reducing the model performance and generalization power. We cope with this issue by reframing WSD as a span extraction problem {---} which we called Extractive Sense Comprehension (ESC) {---} and propose ESCHER, a transformer-based neural architecture for this new formulation. By means of an extensive array of experiments, we show that ESC unleashes the full potential of our model, leading it to outdo all of its competitors and to set a new state of the art on the English WSD task. In the few-shot scenario, ESCHER proves to exploit training data efficiently, attaining the same performance as its closest competitor while relying on almost three times fewer annotations. Furthermore, ESCHER can nimbly combine data annotated with senses from different lexical resources, achieving performances that were previously out of everyone{'}s reach. The model along with data is available at https://github.com/SapienzaNLP/esc.",
}
```


## Environment Setup
To set up the python environment for this project, we strongly suggest using the bash script ```setup.sh``` that 
you can find at top level in this repo. This script will create a new conda environment and take care of all
the requirements and the data needed for the project. Simply run on the command line:
```bash
bash ./setup.sh
```
and follow the instructions.


## Checkpoints
These are the checkpoints of escher when trained on:
- [SemCor](https://drive.google.com/file/d/100jxjLIdmSzrMXXOWgrPz93EG0JBnkfr/view?usp=sharing) (SE07: 76.3 | ALL: 80.7)
- SemCor & Oxford (Available upon request, SE07: 77.8 | ALL: 81.5)

## Prediction and Evaluation
You can disambiguate a corpus using the script ```esc/predict.py```:
```bash
PYTHONPATH=$(pwd) python esc/predict.py --ckpt <escher_checkpoint.ckpt> --dataset-paths data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml --prediction-types probabilistic
```

Where the dataset-paths that you provide to the model must be in a format that follows the one introduced by [Raganato et al. (2017)](https://www.aclweb.org/anthology/E17-1010/).
For reference, all the datasets in the directory ```data/WSD_Evaluation_Framework``` follow this format.
The predictions will be saved in the folder ```predictions``` with the name ```<dataset_name>_predictions.txt```.

If you want to evaluate the model on a dataset, just add the parameter ```--evaluate``` on the previous command.

## Training
If you want to train your own escher model you just have to run the following command:
```bash
PYTHONPATH=$(pwd) python esc/train.py --run_name fresh_escher_model --add_glosses_noise --train_path data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml
```

All the hyperparameters are set by default to the ones utilized in the paper. If you want to 
list them all just execute:
```bash
PYTHONPATH=$(pwd) python esc/train.py -h
```

To parse the hyperparameters in input we use [argparse](https://docs.python.org/3/library/argparse.html),
so it is very simple to change them. For example to modify the learning rate to 0.0005 and the gradient 
accumulation steps to 10 you can execute the following command:
```bash
PYTHONPATH=$(pwd) python esc/train.py --learning_rate 0.0005 --gradient_acc_steps 10 --run_name fresh_escher_model --add_glosses_noise --train_path data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml
```

## License
This project is released under the CC-BY-NC 4.0 license (see license.txt). If you use ESC, please put a link to this repo and cite the paper: [ESC: Redesigning WSD with Extractive Sense Comprehension](https://www.aclweb.org/anthology/2021.naacl-main.371/).

## Acknowledgements
The authors gratefully acknowledge the support of the [ERC Consolidator Grant MOUSSE](http://mousse-project.org) No. 726487 under the European Union's Horizon 2020 research and innovation programme.

This work was supported in part by the MIUR under the grant "Dipartimenti di eccellenza 2018-2022" of the Department of Computer Science of the Sapienza University of Rome.

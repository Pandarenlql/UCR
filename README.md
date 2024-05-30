# A Unified Character-Radical Dual-Supervision Framework for Accurate Chinese Character Recognition with Zero-Shot Learning Support

The official code of UCR (under review). 

UCR effectively incorporate character- and radical-based Chinese character recognition approaches in a unified frawmework. Our proposed framework consists of three parts, including a character recognition module (CRM), a radical recognition module (RRM), and a confidence-based predictor module (CPM), which CRM performs character-based CCR, RRM performs radical-based CCR, and CPM integrates the predictions from two modules. Additionally, two custom modules (global feature aggregation and character feature reconstruction) are devised for different purposes in CRM and RRM, respectively. More importantly, we devise a dual supervision mechanism between CRM and RRM.   The first
supervision builds the cross-modal correspondences between the visual character features and the textual radical features, while the second supervision pulls the reconstructed character feature closer to the holistic feature.   

## Requirements

```pip install -r requirements.txt```

## Training
You can train the UCR by this command:
```python train.py --gpu 0 -p 1500 --dataset <dataset_name>  --data_root <your data path> --batch_size 64```

## Evaluation
Get the pretrained weights from [BaiduNetdisk(passwd:rb9g)](https://pan.baidu.com/s/1KY9AYmjzNv9cipYMWk_RYQ). Two Chinese character Recognition competitions show the strength of our method:

- [CTW](https://codalab.lisn.upsaclay.fr/competitions/13146#results)
- [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=evaluation&task=1)

The recognition results of these two datasets are saved in `./files/ctw-test` and `./files/rects-test`.

To evalute this method, you need prepare the test data according to the tutorial from the  competition page, then you need download the pretrained weights and save it in `./checkpoint`, finally, run the test command:

`python ctw-test.py --gpu 0 --batch_size 1`

or

`python rects-test.py --gpu 0 --batch_size 1`
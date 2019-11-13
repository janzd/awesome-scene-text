[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

# Awesome Scene Text
A curated list of papers and resources for scene text detection and recognition

The year when a paper was first published, including ArXiv publications, is used. As a result, there may be cases when a paper was accepted for example to CVPR 2019, but it is listed in year 2018 because it was published in 2018 on ArXiv.

| Table of contents |
|------------------------------|
| 1. [Scene Text Detection](#scene-text-detection-including-methods-for-end-to-end-detection-and-recognition) |
| 2. [Weakly Supervised Scene Text Detection](#weakly-supervised-scene-text-detection--recognition) |
| 3. [Scene Text Recognition](#scene-text-recognition) |
| 4. [Other scene text papers](#other-scene-text-related-papers) |
| 5. [Scene Text Survey papers](#scene-text-survey) |

## Scene Text Detection (including methods for end-to-end detection and recognition)

### 2010
- Detecting text in natural scenes with stroke width transform [CVPR 2010] [[paper](https://ieeexplore.ieee.org/abstract/document/5540041/)]
- A Method for Text Localization and Recognition in Real-World Images [ACCV 2010] [[paper](https://link.springer.com/chapter/10.1007/978-3-642-19318-7_60)]

### 2011

### 2012
- Real-time scene text localization and recognition [CVPR 2012] [[paper](https://ieeexplore.ieee.org/abstract/document/6248097/)]

### 2013

### 2014
- Robust Scene Text Detection with Convolution Neural Network Induced MSER Trees [ECCV 2014] [[paper](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_33)]

### 2015
- Symmetry-based text line detection in natural scenes [CVPR 2015] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Symmetry-Based_Text_Line_2015_CVPR_paper.pdf)]
- Object proposals for text extraction in the wild [ICDAR 2015] [[paper](https://arxiv.org/abs/1509.02317)]
- Text-Attentional Convolutional Neural Network for Scene Text Detection [TIP 2016] [[paper](https://arxiv.org/abs/1510.03283)]
- Text Flow : A Unified Text Detection System in Natural Scene Images [ICCV 2015] [[paper](https://arxiv.org/abs/1604.06877)]

### 2016
- Accurate Text Localization in Natural Image with Cascaded Convolutional Text Network [ArXiv] [[paper](https://arxiv.org/abs/1603.09423)]
- Multi-Oriented Text Detection With Fully Convolutional Networks [CVPR 2016] [[paper](https://arxiv.org/abs/1604.04018)]
- Scene Text Detection Via Holistic, Multi-Channel Prediction [ArXiv] [[paper](https://arxiv.org/abs/1606.09002)]
- Detecting Text in Natural Image with Connectionist Text Proposal Network [ECCV 2016] [[paper](https://arxiv.org/abs/1609.03605)]
  - https://github.com/tianzhi0549/CTPN [Caffe]
  - https://github.com/eragonruan/text-detection-ctpn [TF]
  - https://github.com/Li-Ming-Fan/OCR-DETECTION-CTPN [TF]
- TextBoxes: A Fast Text Detector with a Single Deep Neural Network [AAAI 2017] [[paper](https://arxiv.org/abs/1611.06779)]
  - https://github.com/MhLiao/TextBoxes [Caffe]
  - https://github.com/shinjayne/shinTB [TF]

### 2017
- Multi-scale FCN with Cascaded Instance Aware Segmentation for Arbitrary Oriented Word Spotting In The Wild [CVPR 2017] [[paper](https://ieeexplore.ieee.org/document/8099541)]
- Deep TextSpotter: An End-To-End Trainable Scene Text Localization and Recognition Framework [ICCV 2017] [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf)]
- Arbitrary-Oriented Scene Text Detection via Rotation Proposals [TMM 2018] [[paper](https://arxiv.org/abs/1703.01086)]
  - https://github.com/mjq11302010044/RRPN [Caffe]
- Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection [CVPR 2017] [[paper](https://arxiv.org/abs/1703.01425)]
- Detecting Oriented Text in Natural Images by Linking Segments [CVPR 2017] [[paper](https://arxiv.org/abs/1703.06520)]
  - https://github.com/bgshih/seglink [TF]
  - https://github.com/dengdan/seglink [TF]
- Deep Direct Regression for Multi-Oriented Scene Text Detection [ICCV 2017] [[paper](https://arxiv.org/abs/1703.08289)]
- Cascaded Segmentation-Detection Networks for Word-Level Text Spotting [ArXiv] [[paper](https://arxiv.org/abs/1704.00834)]
- EAST: An Efficient and Accurate Scene Text Detector [CVPR 2017] [[paper](https://arxiv.org/abs/1704.03155)]
  - https://github.com/argman/EAST [TF]
  - https://github.com/kurapan/EAST [Keras]
- WordFence: Text Detection in Natural Images with Border Awareness [ICIP 2017] [[paper](https://arxiv.org/abs/1705.05483)]
- R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection [ArXiv] [[paper](https://arxiv.org/abs/1706.09579)]
  - https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow [TF]
  - https://github.com/beacandler/R2CNN [Caffe]
- WordSup: Exploiting Word Annotations for Character based Text Detection [ICCV 2017] [[paper](https://arxiv.org/abs/1708.06720)]
- Single Shot Text Detector With Regional Attention [ICCV 2017] [[paper](https://arxiv.org/abs/1709.00138)]
  - https://github.com/BestSonny/SSTD [Caffe]
  - https://github.com/HotaekHan/SSTDNet [PyTorch]
- Fused Text Segmentation Networks for Multi-oriented Scene Text Detection [ArXiv] [[paper](https://arxiv.org/abs/1709.03272)]
- Deep Residual Text Detection Network for Scene Text [ICDAR 2017] [[paper](https://arxiv.org/abs/1711.04147)]
- Feature Enhancement Network: A Refined Scene Text Detector [AAAI 2018] [[paper](https://arxiv.org/abs/1711.04249)]
- ArbiText: Arbitrary-Oriented Text Detection in Unconstrained Scene [ArXiv] [[paper](https://arxiv.org/abs/1711.11249)]
- Self-organized Text Detection with Minimal Post-processing via Border Learning [ICCV 2017] [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf)]
  - https://gitlab.com/rex-yue-wu/ISI-PPT-Text-Detector [Keras]

### 2018
- PixelLink: Detecting Scene Text via Instance Segmentation [AAAI 2018] [[paper](https://arxiv.org/abs/1801.01315)]
  - https://github.com/ZJULearning/pixel_link [TF]
  - https://github.com/BowieHsu/tensorflow_ocr [TF]
- FOTS: Fast Oriented Text Spotting With a Unified Network [CVPR 2018] [[paper](https://arxiv.org/abs/1801.01671)]
- TextBoxes++: A Single-Shot Oriented Scene Text Detector [TIP 2018] [[paper](https://arxiv.org/abs/1801.02765)]
  - https://github.com/MhLiao/TextBoxes_plusplus [Caffe]
- Multi-oriented Scene Text Detection via Corner Localization and Region Segmentation [CVPR 2018] [[paper](https://arxiv.org/abs/1802.08948)]
- An end-to-end TextSpotter with Explicit Alignment and Attention [CVPR 2018] [[paper](https://arxiv.org/abs/1803.03474)]
  - https://github.com/tonghe90/textspotter [Caffe]
- Rotation-Sensitive Regression for Oriented Scene Text Detection [CVPR 2018] [[paper](https://arxiv.org/abs/1803.05265)]
  - https://github.com/MhLiao/RRD [Caffe]
- Detecting multi-oriented text with corner-based region proposals [Neurocomputing 2019] [[paper](https://arxiv.org/abs/1804.02690)]
  - https://github.com/xhzdeng/crpn [Caffe]
- An Anchor-Free Region Proposal Network for Faster R-CNN based Text Detection Approaches [ArXiv] [[paper](https://arxiv.org/abs/1804.09003)]
- IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection [IJCAI 2018] [[paper](https://arxiv.org/abs/1805.01167)]
  - https://github.com/xieyufei1993/InceptText-Tensorflow [TF]
- Shape Robust Text Detection with Progressive Scale Expansion Network [CVPR 2019] [[paper](https://arxiv.org/abs/1806.02559)] [[paper v2](https://arxiv.org/abs/1903.12473)]
  - https://github.com/whai362/PSENet [PyTorch]
  - https://github.com/liuheng92/tensorflow_PSENet [TF]
- TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes [ECCV 2018] [[paper](https://arxiv.org/abs/1807.01544)]
  - https://github.com/princewang1994/TextSnake.pytorch [PyTorch]
- Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes [ECCV 2018] [[paper](https://arxiv.org/abs/1807.02242)]
  - https://github.com/lvpengyuan/masktextspotter.caffe2 [Caffe2]
- Accurate Scene Text Detection through Border Semantics Awareness and Bootstrapping [ECCV 2018] [[paper](https://arxiv.org/abs/1807.03547)]
- A New Anchor-Labeling Method For Oriented Text Detection Using Dense Detection Framework [SPL 2018] [[paper](https://ieeexplore.ieee.org/iel7/97/4358004/08403317.pdf)]
- An Efficient System for Hazy Scene Text Detection using a Deep CNN and Patch-NMS [ICPR 2018] [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545198&tag=1)]
- Scene Text Detection with Supervised Pyramid Context Network [AAAI 2019] [[paper](http://arxiv.org/abs/1811.08605)]
- Pixel-Anchor: A Fast Oriented Scene Text Detector with Combined Networks [ArXiv] [[paper](https://arxiv.org/abs/1811.07432)]
- Mask R-CNN with Pyramid Attention Network for Scene Text Detection [WACV 2019] [[paper](https://arxiv.org/abs/1811.09058)]
- TextMountain: Accurate Scene Text Detection via Instance Segmentation [ArXiv] [[paper](https://arxiv.org/abs/1811.12786)]
- TextField: Learning A Deep Direction Field for Irregular Scene Text Detection [ArXiv] [[paper](https://arxiv.org/abs/1812.01393)]
- TextNet: Irregular Text Reading from Images with an End-to-End Trainable Network [ACCV 2018] [[paper](https://arxiv.org/abs/1812.09900)]

### 2019
- MSR: Multi-Scale Shape Regression for Scene Text Detection [IJCAI 2019] [[paper](https://arxiv.org/abs/1901.02596)]
- Scene Text Detection with Inception Text Proposal Generation Module [ICMLC 2019] [[paper](https://www.researchgate.net/publication/333161163_Scene_Text_Detection_with_Inception_Text_Proposal_Generation_Module)]
- Towards Robust Curve Text Detection with Conditional Spatial Expansion [CVPR 2019] [[paper](https://arxiv.org/abs/1903.08836)]
- Curve Text Detection with Local Segmentation Network and Curve Connection [ArXiv] [[paper](https://arxiv.org/abs/1903.09837)]
- Pyramid Mask Text Detector [ArXiv] [[paper](https://arxiv.org/abs/1903.11800)]
- Tightness-aware Evaluation Protocol for Scene Text Detection [CVPR 2019] [[paper](https://arxiv.org/abs/1904.00813)]
- Character Region Awareness for Text Detection [CVPR 2019] [[paper](https://arxiv.org/abs/1904.01941)]
- Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes [CVPR 2019] [[paper](https://arxiv.org/abs/1904.06535)]
- TextCohesion: Detecting Text for Arbitrary Shapes [ArXiv] [[paper](https://arxiv.org/abs/1904.12640)]
- Arbitrary Shape Scene Text Detection With Adaptive Text Region Representation [CVPR 2019] [[paper](https://arxiv.org/abs/1905.05980)]
- Learning Shape-Aware Embedding for Scene Text Detection [CVPR 2019] [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Learning_Shape-Aware_Embedding_for_Scene_Text_Detection_CVPR_2019_paper.pdf)]
- A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning [ACMMM 2019] [[paper](https://arxiv.org/abs/1908.05498)]
- Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network [ICCV 2019] [[paper](https://arxiv.org/abs/1908.05900)]
- Towards Unconstrained End-to-End Text Spotting [ICCV 2019] [[paper](https://arxiv.org/abs/1908.09231)]
- TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf)]
- Convolutional Character Networks [ICCV 2019] [[paper](https://arxiv.org/abs/1910.07954)]

## Weakly supervised Scene Text Detection & Recognition

### 2017
- Attention-Based Extraction of Structured Information from Street View Imagery [ICDAR 2017] [[paper](https://arxiv.org/abs/1704.03549)]
- WeText: Scene Text Detection under Weak Supervision [ICCV 2017] [[paper](https://arxiv.org/abs/1710.04826)]
- SEE: Towards Semi-Supervised End-to-End Scene Text Recognition [AAAI 2018] [[paper](https://arxiv.org/abs/1712.05404)]
  - https://github.com/Bartzi/see [Chainer]

## Scene Text Recognition

### 2014
- Deep Structured Output Learning for Unconstrained Text Recognition [ICLR 2015] [[paper](https://arxiv.org/abs/1412.5903)]
  - https://github.com/AlexandreSev/Structured_Data [TF]
- Reading text in the wild with convolutional neural networks [IJCV 2016] [[paper](https://arxiv.org/abs/1412.1842)]
  - https://github.com/mathDR/reading-text-in-the-wild [Keras]

### 2015
- Reading Scene Text in Deep Convolutional Sequences [AAAI 2016] [[paper](https://arxiv.org/abs/1506.04395)]
- An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition [TPAMI 2017] [[paper](https://arxiv.org/abs/1507.05717)]
  - https://github.com/bgshih/crnn [Torch]
  - https://github.com/weinman/cnn_lstm_ctc_ocr [TF]
  - https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow [TF]
  - https://github.com/MaybeShewill-CV/CRNN_Tensorflow [TF]
  - https://github.com/meijieru/crnn.pytorch [PyTorch]
  - https://github.com/kurapan/CRNN [Keras]

### 2016
- Recursive Recurrent Nets with Attention Modeling for OCR in the Wild [CVPR 2016] [[paper](https://arxiv.org/abs/1603.03101)]
- Robust scene text recognition with automatic rectification [CVPR 2016] [[paper](https://arxiv.org/abs/1603.03915)]
  - https://github.com/WarBean/tps_stn_pytorch [PyTorch]
  - https://github.com/marvis/ocr_attention [PyTorch]
- CNN-N-Gram for Handwriting Word Recognition [CVPR 2016] [[paper](https://ieeexplore.ieee.org/document/7780622)]
- STAR-Net: A SpaTial Attention Residue Network for Scene Text Recognition [BMVC 2016] [[paper](http://www.bmva.org/bmvc/2016/papers/paper043/paper043.pdf)]

### 2017
- STN-OCR: A single Neural Network for Text Detection and Text Recognition [ArXiv] [[paper](https://arxiv.org/pdf/1707.08831.pdf)]
  - https://github.com/Bartzi/stn-ocr [MXNet]
- Learning to Read Irregular Text with Attention Mechanisms [IJCAI 2017] [[paper](https://www.ijcai.org/proceedings/2017/458)]
- Scene Text Recognition with Sliding Convolutional Character Models [ArXiv] [[paper](https://arxiv.org/abs/1709.01727)]
- Focusing Attention: Towards Accurate Text Recognition in Natural Images [ICCV 2017] [[paper](https://arxiv.org/abs/1709.02054)]
- AON: Towards Arbitrarily-Oriented Text Recognition [CVPR 2018] [[paper](https://arxiv.org/abs/1711.04226)]
  - https://github.com/huizhang0110/AON [TF]
- Gated Recurrent Convolution Neural Network for OCR [NIPS 2017] [[paper](https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)]
  - https://github.com/Jianfeng1991/GRCNN-for-OCR [Torch]

### 2018
- Char-Net: A Character-Aware Neural Network for Distorted Scene Text Recognition [AAAI 2018] [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16327/16307)]
- SqueezedText: A Real-time Scene Text Recognition by Binary Convolutional Encoder-decoder Network [AAAI 2018] [[paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16354)]
- Edit Probability for Scene Text Recognition [CVPR 2018] [[paper](https://arxiv.org/abs/1805.03384)]
- ASTER: An Attentional Scene Text Recognizer with Flexible Rectification [TPAMI 2018] [[paper](https://ieeexplore.ieee.org/document/8395027/)]
  - https://github.com/bgshih/aster [TF]
- Synthetically Supervised Feature Learning for Scene Text Recognition [ECCV 2018] [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.pdf)]
- Scene Text Recognition from Two-Dimensional Perspective [AAAI 2019] [[paper](https://arxiv.org/abs/1809.06508)]
- ESIR: End-to-end Scene Text Recognition via Iterative Image Rectification [CVPR 2019] [[paper](https://arxiv.org/abs/1812.05824)]

### 2019
- A Multi-Object Rectified Attention Network for Scene Text Recognition [Pattern Recognition] [[paper](https://arxiv.org/abs/1901.03003)]
  - https://github.com/Canjie-Luo/MORAN_v2 [PyTorch]
- A Simple and Robust Convolutional-Attention Network for Irregular Text Recognition [[paper](https://arxiv.org/abs/1904.01375)]
- Aggregation Cross-Entropy for Sequence Recognition [CVPR 2019][[paper](https://arxiv.org/abs/1904.08364)]
  - https://github.com/summerlvsong/Aggregation-Cross-Entropy [PyTorch]
- Sequence-to-Sequence Domain Adaptation Network for Robust Text Image Recognition [CVPR 2019][[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Sequence-To-Sequence_Domain_Adaptation_Network_for_Robust_Text_Image_Recognition_CVPR_2019_paper.pdf)]
- 2D Attentional Irregular Scene Text Recognizer [ArXiv] [[paper](https://arxiv.org/abs/1906.05708)]
- Deep Neural Network for Semantic-based Text Recognition in Images [ArXiv] [[paper](https://arxiv.org/abs/1908.01403)]
- Symmetry-constrained Rectification Network for Scene Text Recognition [ICCV 2019] [[paper](https://arxiv.org/abs/1908.01957)]
- Rethinking Irregular Scene Text Recognition (ICDAR 2019-ArT) [[paper](https://arxiv.org/abs/1908.11834)]
  - https://github.com/Jyouhou/ICDAR2019-ArT-Recognition-Alchemy [PyTorch]
- Focus-Enhanced Scene Text Recognition with Deformable Convolutions [ArXiv] [[paper](https://arxiv.org/abs/1908.10998)]
  - https://github.com/Alpaca07/dtr [PyTorch]
- Adaptive Embedding Gate for Attention-Based Scene Text Recognition [ArXiv] [[paper](https://arxiv.org/abs/1908.09475)]

## Script Identification

## Other scene text related papers
### 2016
- Synthetic Data for Text Localisation in Natural Images [CVPR 2016] [[paper](https://arxiv.org/abs/1604.06646)]
  - https://github.com/ankush-me/SynthText

### 2019
- Scene Text Synthesis for Efficient and Effective Deep Network Training [ArXiv] [[paper](http://arxiv.org/abs/1901.09193)]

## Scene text survey
### 2018
- Scene Text Detection and Recognition: The Deep Learning Era [ArXiv] [[paper](https://arxiv.org/abs/1811.04256)]

### 2019
- Scene text detection and recognition with advances in deep learning: a survey [IJDAR 2019] [[paper](https://link.springer.com/article/10.1007%2Fs10032-019-00320-5)]

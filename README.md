  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

  # Awesome Scene Text

  A curated list of papers and resources for scene text detection and recognition. This repository tracks the latest advances in text-in-the-wild research from 2010 to 2025.

  > **Note on Paper Dating**: We use the year when a paper was first publicly available (including arXiv preprints) rather than the conference publication year. For example, a paper published on arXiv in 2023 but accepted to CVPR 2024 is listed under 2023.

  ## Statistics

  - **Total Papers**: 160+
  - **Timespan**: 2010-2025
  - **Focus Areas**: Detection, Recognition, End-to-End Spotting, Video Text, Generation, and more

  ## Table of Contents

  1. [Survey Papers](#survey-papers)
  2. [Tools & Libraries](#tools--libraries)
  3. [Datasets & Benchmarks](#datasets--benchmarks)
  4. [Scene Text Detection](#scene-text-detection)
  5. [Scene Text Recognition](#scene-text-recognition)
  6. [End-to-End Text Spotting](#end-to-end-text-spotting)
  7. [Video Text Detection & Recognition](#video-text-detection--recognition)
  8. [Text Generation with Diffusion Models](#text-generation-with-diffusion-models)
  9. [Text Editing & Removal](#text-editing--removal)
  10. [Weakly Supervised Methods](#weakly-supervised-methods)
  11. [Multilingual & Low-Resource Languages](#multilingual--low-resource-languages)
  12. [Document AI & Layout Analysis](#document-ai--layout-analysis)
  13. [Other Scene Text Papers](#other-scene-text-papers)

  ---

  ## Survey Papers

  ### 2025
  - **Challenges and Gaps in Scene Text Detection and Recognition: A Detailed Survey** [[paper](https://link.springer.com/chapter/10.1007/978-981-96-2694-6_26)]
  - **Self-Supervised Learning for Text Recognition: A Critical Survey** [IJCV 2025] [[paper](https://link.springer.com/article/10.1007/s11263-025-02487-3)]
  - **Handwritten Text Recognition: A Survey** [arXiv 2025] [[paper](https://arxiv.org/abs/2502.08417v1)]

  ### 2024
  - **A Comprehensive Survey of Transformers in Text Recognition: Techniques, Challenges, and Future Directions** [ACM Computing Surveys 2024] [[paper](https://dl.acm.org/doi/10.1145/3771273)]

  ### 2023
  - **Scene text understanding: recapitulating the past decade** [Artificial Intelligence Review 2023] [[paper](https://link.springer.com/article/10.1007/s10462-023-10530-3)]

  ### 2022
  - **Scene text detection and recognition: a survey** [Multimedia Tools and Applications 2022] [[paper](https://link.springer.com/article/10.1007/s11042-022-12693-7)]
  - **A survey on methods, datasets and implementations for scene text spotting** [IET Image Processing 2022] [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12574)]

  ### 2019
  - **Scene Text Detection and Recognition: The Deep Learning Era** [arXiv 2019] [[paper](https://arxiv.org/abs/1811.04256)]

  ### 2018
  - **Scene text detection and recognition with advances in deep learning: a survey** [IJDAR 2019] [[paper](https://link.springer.com/article/10.1007%2Fs10032-019-00320-5)]

  ---

  ## Tools & Libraries

  ### Open Source OCR Systems

  - **PaddleOCR** - Powerful, lightweight OCR toolkit supporting 100+ languages [[code](https://github.com/PaddlePaddle/PaddleOCR)]
  - **EasyOCR** - Ready-to-use OCR with 80+ languages support (PyTorch-based) [[code](https://github.com/JaidedAI/EasyOCR)]
  - **MMOCR** - Comprehensive OCR toolbox with 7 detection and 5 recognition algorithms [[code](https://github.com/open-mmlab/mmocr)]
  - **OpenOCR** - Unified benchmark system for training and evaluating scene text models [[code](https://github.com/Topdu/OpenOCR)]

  ### Comparison
  For detailed comparisons of these tools, see:
  - [OCR comparison: Tesseract vs EasyOCR vs PaddleOCR vs MMOCR](https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66)
  - [Open-Source OCR Libraries: A Comprehensive Study](https://aclanthology.org/2024.icon-1.48.pdf)

  ---

  ## Datasets & Benchmarks

  ### Major Datasets

  #### ICDAR Datasets (Standard Benchmarks)

  **Horizontal Text:**
  - **ICDAR 2003 (IC03)** - 509 images (258 train, 251 test), 2,266 text instances, English only [[download](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions)]
  - **ICDAR 2013 (IC13)** - 462 images (229 train, 233 test), 1,944 text instances [[download](https://rrc.cvc.uab.es/?ch=2)]
  - **ICDAR 2015 (IC15)** - 1,500 images (1,000 train, 500 test), 17,548 text instances, first incidental scene text dataset [[download](https://rrc.cvc.uab.es/?ch=4)]

  **Multi-Lingual Text:**
  - **ICDAR 2017 MLT** - 10,000 images, 9 languages, word-level annotations [[paper](https://ieeexplore.ieee.org/document/8270168/)] [[download](https://rrc.cvc.uab.es/?ch=8)]
  - **ICDAR 2019 MLT** - 20,000 images, 10 languages, word-level annotations [[paper](https://arxiv.org/abs/1907.00945)] [[download](https://rrc.cvc.uab.es/?ch=15)]

  **Arbitrary-Shaped Text:**
  - **ICDAR 2019 ArT** - 10,166 images (5,603 train, 4,563 test), diverse text shapes [[download](https://rrc.cvc.uab.es/?ch=14)]

  **Chinese Text:**
  - **ICDAR 2017 RCTW-17** - 12,514 images (11,514 train, 1,000 test), English/Chinese [[download](http://rctw.vlrlab.net/dataset/)]
  - **ICDAR 2019 ReCTS** - 20,000 images, Chinese street view trademark dataset [[download](https://rrc.cvc.uab.es/?ch=12)]

  #### Common Benchmark Datasets

  **Horizontal/Multi-Oriented Text:**
  - **COCO-Text** - 63,686 images (43,686 train, 20,000 test), 145,859 text instances, multilingual [[paper](https://arxiv.org/abs/1601.07140)] [[download](https://bgshih.github.io/cocotext/)]
  - **MSRA-TD500** - 500 images (300 train, 200 test), English/Chinese, text-line level [[download](https://huggingface.co/datasets/yunusserhat/MSRA-TD500-Dataset)]
  - **SVT (Street View Text)** - 350 images, 725 text instances [[download](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)]
  - **USTB-SV1K** - 1,000 street view images, 2,955 text instances [[download](https://github.com/onionych/USTB-SV1K)]
  - **IIIT5K** - 5,000 word images (2,000 train, 3,000 test), 50-word and 1,000-word lexicons [[download](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)]

  **Curved/Irregular Text:**
  - **Total-Text** - 1,555 images, 11,459 text instances, horizontal/multi-oriented/curved [[paper](https://arxiv.org/abs/1710.10400)] [[download](https://github.com/cs-chan/Total-Text-Dataset)]
  - **SCUT-CTW1500** - 1,500 images (1,000 train, 500 test), 10,751 text instances, 14-vertex polygon annotations [[download](https://github.com/Yuliang-Liu/Curve-Text-Detector)]
  - **CUTE80** - 80 high-resolution images, 288 cropped curved text instances [[download](http://cs-chan.com/downloads_CUTE80_dataset.html)]
  - **LSVT** - 450,000 images (430,000 train, 20,000 test), horizontal/multi-oriented/curved text [[download](https://rrc.cvc.uab.es/?ch=16)]

  **Chinese Text:**
  - **CTW (Chinese Text in the Wild)** - 32,285 images, 1,018,402 character instances, character-level with 6 attributes [[download](https://ctwdataset.github.io/)]

  **Synthetic Datasets:**
  - **SynthText** - 800,000 images, 6 million text instances [[code](https://github.com/ankush-me/SynthText)]
  - **Synth80k** - 800,000 images, ~8 million synthetic word instances [[download](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)]

  #### Large-Scale Modern Datasets
  - **Union14M** (2023) - 4M labeled + 10M unlabeled images for STR [[paper](https://arxiv.org/abs/2307.08723)] [[download](https://github.com/Mountchicken/Union14M)]
  - **HierText** (ICDAR 2023) - Hierarchical text with word/line/paragraph annotations, 103.8 words/image [[download](https://research.google/blog/announcing-the-icdar-2023-competition-on-hierarchical-text-detection-and-recognition/)]
  - **TextOCR** - 900k annotated words on real images [[paper](https://arxiv.org/abs/2105.05486)] [[download](https://textvqa.org/textocr/)]
  - **OCRBench v2** (2025) - Comprehensive benchmark for LMMs across 8 text-oriented abilities [[paper](https://arxiv.org/abs/2501.00321)] [[download](https://99franklin.github.io/ocrbench_v2/)]
  - **WordArt-V1.5** (ICDAR 2024) - 12,000 artistic text images [[paper](https://link.springer.com/chapter/10.1007/978-3-031-70552-6_18)] [[download](https://sites.google.com/view/icdar-2024-competition-wordart/)]

  #### Video Text
  - **DSText** (ICDAR 2023) - 100 video clips from 12 scenarios [[paper](https://arxiv.org/abs/2304.04376)] [[download](https://rrc.cvc.uab.es/?ch=22)]
  - **BOVText** (2021) - Bilingual, OpenWorld video text with 2,000+ videos, 1.75M+ frames [[paper](https://arxiv.org/abs/2112.04888)] [[download](https://github.com/weijiawu/BOVText-Benchmark)]

  #### Specialized Domains

  **Visual Question Answering (VQA) with Scene Text:**
  - **TextVQA** - 45,336 questions on 28,408 images requiring reading and reasoning about text [[paper](https://arxiv.org/abs/1904.08920)] [[website](https://textvqa.org/)] [[dataset](https://huggingface.co/datasets/facebook/textvqa)]
  - **ST-VQA** (Scene Text VQA) - 23,038 images, 31,791 QA pairs, all questions require scene text reading [[paper](https://arxiv.org/abs/1905.13648)] [[download](https://rrc.cvc.uab.es/?ch=11)]
  - **OCR-VQA** - 200k+ QA pairs on book cover images [[paper](https://anandmishra22.github.io/files/mishra-OCR-VQA.pdf)] [[download](https://ocr-vqa.github.io/)]
  - **DocVQA** - 50,000 questions on 12,000+ document images [[paper](https://arxiv.org/abs/2007.00398)] [[website](https://www.docvqa.org/)] [[download](https://rrc.cvc.uab.es/?ch=17)]
  - **InfographicsVQA** - 5,000+ infographic images, 30,000 QA pairs [[paper](https://arxiv.org/abs/2104.12756)] [[website](https://www.docvqa.org/datasets/infographicvqa)]
  - **TextCaps** - 145k captions for 28k images requiring text reading and reasoning [[paper](https://arxiv.org/abs/2003.12462)] [[download](https://textvqa.org/textcaps/)]
  - **ViTextVQA** - Vietnamese text comprehension in images [[paper](https://arxiv.org/abs/2404.10652)] [[download](https://github.com/minhquan6203/ViTextVQA-Dataset)]
  - **ViOCRVQA** - 28,000+ images, 120,000+ Vietnamese QA pairs [[paper](https://arxiv.org/abs/2404.18397v1)] [[download](https://github.com/qhnhynmm/ViOCRVQA-Dataset)]

  **Mathematical Expressions:**
  - **MathWriting** (NeurIPS 2023) - Handwritten mathematical expressions [[paper](https://arxiv.org/pdf/2404.10690)]

  **Other Domains:**
  - **MapText** (ICDAR 2024) - Historical map text detection [[paper](https://link.springer.com/chapter/10.1007/978-3-031-70552-6_22)] [[download](https://rrc.cvc.uab.es/?ch=32)]

  ### Benchmark Repositories
  - **Robust Reading Competition** - Ongoing challenges since 2013 [[website](https://rrc.cvc.uab.es/)]

  ---

  ## Scene Text Detection

  ### 2025
  - **Bharat Scene Text: A Novel Comprehensive Dataset and Benchmark for Indian Language Scene Text Understanding** [arXiv 2025] [[paper](https://arxiv.org/abs/2511.23071)]
  - **The Devil is in Fine-tuning and Long-tailed Problems: A New Benchmark for Scene Text Detection** [IJCAI 2025] [[paper](https://arxiv.org/abs/2505.15649)] [[code](https://github.com/pd162/LTB)]
  - **Scene Text Detection and Recognition "in light of" Challenging Environmental Conditions using Aria Glasses Egocentric Vision Cameras** [arXiv 2025] [[paper](https://arxiv.org/abs/2507.16330)] [[code](https://github.com/josepDe/Project_Aria_STR)]
  - **A Large-scale Dataset for Robust Complex Anime Scene Text Detection** [arXiv 2025] [[paper](https://arxiv.org/abs/2510.07951)] [[dataset](https://huggingface.co/datasets/deepghs/AnimeText)]
  - **Masked Text Pre-Training for Scene Text Detection** [Transactions on Multimedia 2025] [[paper](https://ieeexplore.ieee.org/abstract/document/11175522)]
  - **TextMamba: Scene Text Detector with Mamba** [arXiv 2025] [[paper](https://arxiv.org/abs/2512.06657)]
  - **Occluded scene text detection via context-awareness from sketch-level image representations** [Multimedia Systems 2025] [[paper](https://link.springer.com/article/10.1007/s00530-025-01782-w)]
  - **ContraText-DETR: Boosting Industrial Scene Text Detection Based on Contrastive Learning and Synthetic Low-Contrast Text** [Sensors Journal 2025] [[paper](https://ieeexplore.ieee.org/abstract/document/11062481)]


  ### 2024
  - **Explicit Relational Reasoning Network for Scene Text Detection** [AAAI 2025] [[paper](https://arxiv.org/abs/2412.14692)]
  - **InstructOCR: Instruction Boosting Scene Text Spotting** [AAAI 2025] [[paper](https://arxiv.org/abs/2412.15523v1)]
  - **Arbitrary Reading Order Scene Text Spotter with Local Semantics Guidance** [AAAI 2025] [[paper](https://arxiv.org/abs/2412.10159v1)]
  - **Revisiting Tampered Scene Text Detection in the Era of Generative AI** [AAAI 2025] [[paper](https://arxiv.org/abs/2407.21422v2)] [[code](https://github.com/qcf-568/OSTF)]
  - **TextBlockV2: Towards Precise-Detection-Free Scene Text Spotting with Pre-trained Language Model** [arXiv 2024] [[paper](https://arxiv.org/abs/2403.10047v1)]
  - **ODM: A Text-Image Further Alignment Pre-training Approach for Scene Text Detection and Spotting** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Duan_ODM_A_Text-Image_Further_Alignment_Pre-training_Approach_for_Scene_Text_CVPR_2024_paper.pdf)] [[code](https://github.com/PriNing/ODM)] [[code](https://github.com/PriNing/ODM)]
  - **Bridging the Gap Between End-to-End and Two-Step Text Spotting** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Bridging_the_Gap_Between_End-to-End_and_Two-Step_Text_Spotting_CVPR_2024_paper.pdf)] [[code](https://github.com/mxin262/Bridging-Text-Spotting)]
  - **LayoutFormer: Hierarchical Text Detection Towards Scene Text Understanding** [CVPR 2024] [[paper](https://cvpr.thecvf.com/virtual/2024/poster/29240)]
  - **Text Grouping Adapter: Adapting Pre-trained Text Detector for Layout Analysis** [CVPR 2024] [[paper](https://arxiv.org/abs/2405.07481)]
  - **ACP-Net: Asymmetric Center Positioning Network for Real-Time Text Detection** [Knowledge-Based Systems 2024] [[paper](papers.ssrn.com/sol3/papers.cfm?abstract_id=4855396)]
  - **GridMask: An Efficient Scheme for Real Time Curved Scene Text Detection** [[paper](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_11)]
  - **TextBlockV2: Towards Precise-Detection-Free Scene Text Spotting with Pre-trained Language Model** [Transactions on Multimedia Computing, Communications and Applications 2025] [[paper](https://arxiv.org/abs/2403.10047)]
  - **SwinTextSpotter v2: Towards Better Synergy for Scene Text Spotting** [IJCV 2025] [[paper](https://arxiv.org/abs/2401.07641)] [[code](https://github.com/mxin262/SwinTextSpotterv2)]

  ### 2023
  - **LRANet: Towards Accurate and Efficient Scene Text Detection with Low-Rank Approximation Network** [AAAI 2024] [[paper](https://arxiv.org/abs/2306.15142)] [[code](https://github.com/ychensu/LRANet)]
  - **Bridging Synthetic and Real Worlds for Pre-Training Scene Text Detectors** [ECCV 2024] [[paper](https://arxiv.org/pdf/2312.05286)] [[code](https://github.com/SJTU-DeepVisionLab/FreeReal)] [[code](https://github.com/ychensu/LRANet)]
  - **Towards Robust Real-Time Scene Text Detection: From Semantic to Instance Representation Learning** [ACMMM 2023] [[paper](https://arxiv.org/abs/2308.07202)]
  - **DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Multilingual Text Spotting** [CVPR 2023] [[paper](https://arxiv.org/abs/2305.19957)] [[code](https://github.com/ViTAE-Transformer/DeepSolo)]
  - **ESTextSpotter: Towards Better Scene Text Spotting with Explicit Synergy in Transformer** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_ESTextSpotter_Towards_Better_Scene_Text_Spotting_with_Explicit_Synergy_in_ICCV_2023_paper.pdf)] [[code](https://github.com/mxin262/ESTextSpotter)]
  - **Towards Robust Tampered Text Detection in Document Image: New dataset and New Solution** [CVPR 2023] [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.pdf)] [[code](https://github.com/qcf-568/DocTamper)]
  - **Self-Supervised Implicit Glyph Attention for Text Recognition** [CVPR 2023] [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guan_Self-Supervised_Implicit_Glyph_Attention_for_Text_Recognition_CVPR_2023_paper.pdf)] [[code](https://github.com/TongkunGuan/SIGA)]
  - **Arbitrary-shaped scene text detection with keypoint-based shape representation** [IJDAR 2023] [[paper](https://link.springer.com/article/10.1007/s10032-022-00396-6)]
  - **Arbitrary-Shaped Text Detection with B-Spline Curve Network** [Sensors 2023] [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10007203/)]

  ### 2022
  - **SwinTextSpotter: Scene Text Spotting via Better Synergy Between Text Detection and Recognition** [CVPR 2022] [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_SwinTextSpotter_Scene_Text_Spotting_via_Better_Synergy_Between_Text_Detection_CVPR_2022_paper.html)] [[code](https://github.com/mxin262/SwinTextSpotter)]
  - **Few Could Be Better Than All: Feature Sampling and Grouping for Scene Text Detection** [CVPR 2022] [[paper](https://arxiv.org/abs/2203.15221)]
  - **Vision-Language Pre-Training for Boosting Scene Text Detectors** [CVPR 2022] [[paper](https://arxiv.org/abs/2204.13867)]
  - **Towards End-to-End Unified Scene Text Detection and Layout Analysis** [CVPR 2022] [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Long_Towards_End-to-End_Unified_Scene_Text_Detection_and_Layout_Analysis_CVPR_2022_paper.pdf)] [[code](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector)]
  - **TESTR: Text Spotting Transformers** [CVPR 2022] [[paper](https://arxiv.org/abs/2204.01918)]
  - **Contextual Text Block Detection towards Scene Text Understanding** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880371.pdf)]
  - **GLASS: Global to Local Attention for Scene-Text Spotting** [ECCV 2022] [[paper](https://arxiv.org/abs/2208.03364)] [[code](https://github.com/amazon-science/glass-text-spotting)]
  - **Language Matters: A Weakly Supervised Vision-Language Pre-training Approach for Scene Text Detection and Spotting** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880282.pdf)] [[code](https://github.com/bytedance/oclip)]
  - **Arbitrary Shape Text Detection via Boundary Transformer** [Transactions on Multimedia 2023] [[paper](https://arxiv.org/abs/2205.05320)] [[code](https://github.com/GXYM/TextBPN-Plus-Plus)]
  - **Arbitrary shape scene text detector with accurate text instance generation based on instance-relevant contexts** [Multimedia Tools and Applications 2022] [[paper](https://dl.acm.org/doi/10.1007/s11042-022-13897-7)]
  - **Arbitrary Shape Text Detection using Transformers** [arXiv 2022] [[paper](https://arxiv.org/abs/2202.11221)]
  - **Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion** [TPAMI 2022] [[paper](https://arxiv.org/abs/2202.10304)] [[code](https://github.com/MhLiao/DB)]

  ### 2021
  - **Progressive Contour Regression for Arbitrary-Shape Scene Text Detection** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Progressive_Contour_Regression_for_Arbitrary-Shape_Scene_Text_Detection_CVPR_2021_paper.pdf)]
  - **Fourier Contour Embedding for Arbitrary-Shaped Text Detection** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Fourier_Contour_Embedding_for_Arbitrary-Shaped_Text_Detection_CVPR_2021_paper.pdf)]
  - **A Straightforward and Efficient Instance-Aware Curved Text Detector** [Sensors 2021] [[paper](https://www.mdpi.com/1424-8220/21/6/1945)]
  - **FAST: Searching for a Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation** [arXiv 2021] [[paper](https://arxiv.org/abs/2111.02394)] [[code](https://github.com/czczup/FAST)]
  - **Detection and rectification of arbitrary shaped scene texts by using text keypoints and links** [Pattern Recognition 2021] [[paper](https://arxiv.org/abs/2103.00785)]
  - **Arbitrary-shaped scene text detection by predicting distance map** [Applied Intelligence 2021] [[paper](https://dl.acm.org/doi/abs/10.1007/s10489-021-03065-z)]

  ### 2020
  - **UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World** [CVPR 2020] [[paper](https://arxiv.org/abs/2003.10608)] [[code](https://github.com/Jyouhou/UnrealText)]
  - **Character Region Awareness for Text Detection** [CVPR 2019] [[paper](https://arxiv.org/abs/1904.01941)]
  - **Curved scene text detection via transverse and longitudinal sequence connection** [Pattern Recognition 2020] [[paper](https://www.sciencedirect.com/science/article/pii/S0031320319300664)]

  ### 2019
  - **DBNet: Real-time Scene Text Detection with Differentiable Binarization** [AAAI 2020] [[paper](https://arxiv.org/abs/1911.08947)] [[code](https://github.com/MhLiao/DB)]
  - **MSR: Multi-Scale Shape Regression for Scene Text Detection** [IJCAI 2019] [[paper](https://arxiv.org/abs/1901.02596)]
  - **Scene Text Detection with Inception Text Proposal Generation Module** [ICMLC 2019] [[paper](https://dl.acm.org/doi/10.1145/3318299.3318373)]
  - **Towards Robust Curve Text Detection with Conditional Spatial Expansion** [CVPR 2019] [[paper](https://arxiv.org/abs/1903.08836)]
  - **Detecting Curve Text with Local Segmentation Network and Curve Connection** [arXiv 2019] [[paper](https://arxiv.org/abs/1903.09837)]
  - **Pyramid Mask Text Detector** [arXiv 2019] [[paper](https://arxiv.org/abs/1903.11800)]
  - **Tightness-aware Evaluation Protocol for Scene Text Detection** [CVPR 2019] [[paper](https://arxiv.org/abs/1904.00813)] [[code](https://github.com/Yuliang-Liu/TIoU-metric)]
  - **Character Region Awareness for Text Detection** [CVPR 2019] [[paper](https://arxiv.org/abs/1904.01941)] [[code](https://github.com/clovaai/CRAFT-pytorch)]
  - **Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes** [CVPR 2019] [[paper](https://arxiv.org/abs/1904.06535)]
  - **TextCohesion: Detecting Text for Arbitrary Shapes** [arXiv 2019] [[paper](https://arxiv.org/abs/1904.12640)]
  - **Arbitrary Shape Scene Text Detection With Adaptive Text Region Representation** [CVPR 2019] [[paper](https://arxiv.org/abs/1905.05980)]
  - **Learning Shape-Aware Embedding for Scene Text Detection** [CVPR 2019] [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Learning_Shape-Aware_Embedding_for_Scene_Text_Detection_CVPR_2019_paper.pdf)]
  - **A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning** [ACMMM 2019] [[paper](https://arxiv.org/abs/1908.05498)]
  - **Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network** [ICCV 2019] [[paper](https://arxiv.org/abs/1908.05900)] [[code](https://github.com/WenmuZhou/PAN.pytorch)]
  - **Towards Unconstrained End-to-End Text Spotting** [ICCV 2019] [[paper](https://arxiv.org/abs/1908.09231)]
  - **TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting** [ICCV 2019] [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf)]
  - **Convolutional Character Networks** [ICCV 2019] [[paper](https://arxiv.org/abs/1910.07954)] [[code](https://github.com/msight-tech/research-charnet)]

  ### 2018
  - **PixelLink: Detecting Scene Text via Instance Segmentation** [AAAI 2018] [[paper](https://arxiv.org/abs/1801.01315)] [[code](https://github.com/ZJULearning/pixel_link)]
  - **FOTS: Fast Oriented Text Spotting With a Unified Network** [CVPR 2018] [[paper](https://arxiv.org/abs/1801.01671)]
  - **TextBoxes++: A Single-Shot Oriented Scene Text Detector** [TIP 2018] [[paper](https://arxiv.org/abs/1801.02765)] [[code](https://github.com/MhLiao/TextBoxes_plusplus)]
  - **Multi-oriented Scene Text Detection via Corner Localization and Region Segmentation** [CVPR 2018] [[paper](https://arxiv.org/abs/1802.08948)]
  - **An end-to-end TextSpotter with Explicit Alignment and Attention** [CVPR 2018] [[paper](https://arxiv.org/abs/1803.03474)] [[code](https://github.com/tonghe90/textspotter)]
  - **Rotation-Sensitive Regression for Oriented Scene Text Detection** [CVPR 2018] [[paper](https://arxiv.org/abs/1803.05265)] [[code](https://github.com/MhLiao/RRD)]
  - **Detecting multi-oriented text with corner-based region proposals** [Neurocomputing 2019] [[paper](https://arxiv.org/abs/1804.02690)] [[code](https://github.com/xhzdeng/crpn)]
  - **An Anchor-Free Region Proposal Network for Faster R-CNN based Text Detection Approaches** [arXiv 2018] [[paper](https://arxiv.org/abs/1804.09003)]
  - **IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection** [IJCAI 2018] [[paper](https://arxiv.org/abs/1805.01167)] [[code](https://github.com/FakerYFX/InceptText-Tensorflow)]
  - **Shape Robust Text Detection with Progressive Scale Expansion Network** [CVPR 2019] [[paper](https://arxiv.org/abs/1806.02559)] [[paper](https://arxiv.org/abs/1903.12473)] [[code](https://github.com/whai362/PSENet)]
  - **TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes** [ECCV 2018] [[paper](https://arxiv.org/abs/1807.01544)] [[code](https://github.com/princewang1994/TextSnake.pytorch)]
  - **Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes** [ECCV 2018] [[paper](https://arxiv.org/abs/1807.02242)] [[code](https://github.com/MhLiao/MaskTextSpotter)]
  - **Accurate Scene Text Detection through Border Semantics Awareness and Bootstrapping** [ECCV 2018] [[paper](https://arxiv.org/abs/1807.03547)]
  - **A New Anchor-Labeling Method For Oriented Text Detection Using Dense Detection Framework** [SPL 2018] [[paper](https://ieeexplore.ieee.org/iel7/97/4358004/08403317.pdf)]
  - **An Efficient System for Hazy Scene Text Detection using a Deep CNN and Patch-NMS** [ICPR 2018] [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545198&tag=1)]
  - **Scene Text Detection with Supervised Pyramid Context Network** [AAAI 2019] [[paper](http://arxiv.org/abs/1811.08605)]
  - **Pixel-Anchor: A Fast Oriented Scene Text Detector with Combined Networks** [arXiv] [[paper](https://arxiv.org/abs/1811.07432)]
  - **Mask R-CNN with Pyramid Attention Network for Scene Text Detection** [WACV 2019] [[paper](https://arxiv.org/abs/1811.09058)]
  - **TextMountain: Accurate Scene Text Detection via Instance Segmentation** [arXiv] [[paper](https://arxiv.org/abs/1811.12786)]
  - **TextField: Learning A Deep Direction Field for Irregular Scene Text Detection** [arXiv 2018] [[paper](https://arxiv.org/abs/1812.01393)] [[code](https://github.com/YukangWang/TextField)]
  - **TextNet: Irregular Text Reading from Images with an End-to-End Trainable Network** [ACCV 2018] [[paper](https://arxiv.org/abs/1812.09900)]

  ### 2017
  - **Multi-scale FCN with Cascaded Instance Aware Segmentation for Arbitrary Oriented Word Spotting In The Wild** [CVPR 2017] [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/He_Multi-Scale_FCN_With_CVPR_2017_paper.pdf)]
  - **Deep TextSpotter: An End-To-End Trainable Scene Text Localization and Recognition Framework** [ICCV 2017] [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf)]
  - **Arbitrary-Oriented Scene Text Detection via Rotation Proposals** [TMM 2018] [[paper](https://arxiv.org/abs/1703.01086)] [[code](https://github.com/mjq11302010044/RRPN)]
  - **Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection** [CVPR 2017] [[paper](https://arxiv.org/abs/1703.01425)]
  - **Detecting Oriented Text in Natural Images by Linking Segments** [CVPR 2017] [[paper](https://arxiv.org/abs/1703.06520)] [[code](https://github.com/bgshih/seglink)]
  - **Deep Direct Regression for Multi-Oriented Scene Text Detection** [ICCV 2017] [[paper](https://arxiv.org/abs/1703.08289)]
  - **Cascaded Segmentation-Detection Networks for Word-Level Text Spotting** [arXiv 2017] [[paper](https://arxiv.org/abs/1704.00834)]
  - **EAST: An Efficient and Accurate Scene Text Detector** [CVPR 2017] [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf)] [[code - TF](https://github.com/argman/EAST)] [[code - Keras](https://github.com/janzd/EAST)]
  - **WordFence: Text Detection in Natural Images with Border Awareness** [ICIP 2017] [[paper](https://arxiv.org/abs/1705.05483)]
  - **R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection** [arXiv 2017] [[paper](https://arxiv.org/abs/1706.09579)] [[code](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow)]
  - **WordSup: Exploiting Word Annotations for Character based Text Detection** [ICCV 2017] [[paper](https://arxiv.org/abs/1708.06720)]
  - **Single Shot Text Detector With Regional Attention** [ICCV 2017] [[paper](https://arxiv.org/abs/1709.00138)] [[code](https://github.com/HotaekHan/SSTDNet)]
  - **Fused Text Segmentation Networks for Multi-oriented Scene Text Detection** [ICPR 2018] [[paper](https://arxiv.org/abs/1709.03272)]
  - **Deep Residual Text Detection Network for Scene Text** [ICDAR 2017] [[paper](https://arxiv.org/abs/1711.04147)]
  - **Feature Enhancement Network: A Refined Scene Text Detector** [AAAI 2018] [[paper](https://arxiv.org/abs/1711.04249)]
  - **ArbiText: Arbitrary-Oriented Text Detection in Unconstrained Scene** [arXiv 2017] [[paper](https://arxiv.org/abs/1711.11249)]
  - **Self-organized Text Detection with Minimal Post-processing via Border Learning** [ICCV 2017] [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf)] [[code](https://gitlab.com/rex-yue-wu/ISI-PPT-Text-Detector)]

  ### 2016
  - **Accurate Text Localization in Natural Image with Cascaded Convolutional Text Network** [arXiv 2016] [[paper](https://arxiv.org/abs/1603.09423)]
  - **Multi-Oriented Text Detection With Fully Convolutional Networks** [CVPR 2016] [[paper](https://arxiv.org/abs/1604.04018)]
  - **Scene Text Detection Via Holistic, Multi-Channel Prediction** [arXiv 2016] [[paper](https://arxiv.org/abs/1606.09002)]
  - **Detecting Text in Natural Image with Connectionist Text Proposal Network** [ECCV 2016] [[paper](https://arxiv.org/abs/1609.03605)] [[code](https://github.com/eragonruan/text-detection-ctpn)]
  - **TextBoxes: A Fast Text Detector with a Single Deep Neural Network** [AAAI 2017] [[paper](https://arxiv.org/abs/1611.06779)] [[code](https://github.com/MhLiao/TextBoxes)]

  ### 2015
  - **Symmetry-based text line detection in natural scenes** [CVPR 2015] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Symmetry-Based_Text_Line_2015_CVPR_paper.pdf)]
  - **Object proposals for text extraction in the wild** [ICDAR 2015] [[paper](https://arxiv.org/abs/1509.02317)]
  - **Text-Attentional Convolutional Neural Network for Scene Text Detection** [TIP 2016] [[paper](https://arxiv.org/abs/1510.03283)]
  - **Text Flow : A Unified Text Detection System in Natural Scene Images** [ICCV 2015] [[paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Tian_Text_Flow_A_ICCV_2015_paper.pdf)]

  ### 2014
  - **Robust Scene Text Detection with Convolution Neural Network Induced MSER Trees** [ECCV 2014] [[paper](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_33)]

  ### 2012
  - **Real-time scene text localization and recognition** [CVPR 2012] [[paper](https://ieeexplore.ieee.org/abstract/document/6248097/)]

  ### 2010
  - **Detecting text in natural scenes with stroke width transform** [CVPR 2010] [[paper](https://ieeexplore.ieee.org/abstract/document/5540041/)]
  - **A Method for Text Localization and Recognition in Real-World Images** [ACCV 2010] [[paper](https://link.springer.com/chapter/10.1007/978-3-642-19318-7_60)]

  ---

  ## Scene Text Recognition

  ### 2025
  - **HunyuanOCR: Commercial-Grade OCR Vision-Language Model** [arXiv 2025] [[paper](https://arxiv.org/abs/2511.19575)]
  - **A Context-Driven Training-Free Network for Lightweight Scene Text Segmentation and Recognition** [arXiv 2025] [[paper](https://arxiv.org/abs/2503.15639)]
  - **SSCD: Self-Supervised Coherence Discrimination Representation Learning for Scene Text Recognition** [ICMR 2025] [[paper](https://dl.acm.org/doi/10.1145/3731715.3733436)]
  - **CLIP is Almost All You Need: Towards Parameter-Efficient Scene Text Retrieval without OCR** [CVPR 2025] [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_CLIP_is_Almost_All_You_Need_Towards_Parameter-Efficient_Scene_Text_CVPR_2025_paper.pdf)]

  ### 2024
  - **OTE: Exploring Accurate Scene Text Recognition Using One Token** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_OTE_Exploring_Accurate_Scene_Text_Recognition_Using_One_Token_CVPR_2024_paper.pdf)]
  - **Choose What You Need: Disentangled Representation Learning for Scene Text Recognition, Removal and Editing** [CVPR 2024]
  - **An Empirical Study of Scaling Law for Scene Text Recognition** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Rang_An_Empirical_Study_of_Scaling_Law_for_Scene_Text_Recognition_CVPR_2024_paper.pdf)]
  - **VL-Reader: Vision and Language Reconstructor is an Effective Scene Text Recognizer** [arXiv 2024] [[paper](https://arxiv.org/html/2409.11656v1)]
  - **Decoder Pre-Training with only Text for Scene Text Recognition** [arXiv 2024] [[paper](https://arxiv.org/html/2408.05706v1)]
  - **SVIPTR: Fast and Efficient Scene Text Recognition with Vision Permutable Extractor** [arXiv 2024] [[paper](https://arxiv.org/html/2401.10110v4)]
  - **TextViTCNN: Enhancing Natural Scene Text Recognition with Hybrid Transformer and Convolutional Networks** [[paper](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_19)]
  - **Free Lunch: Frame-level Contrastive Learning with Text Perceiver for Robust Scene Text Recognition in Lightweight Models** [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681045)]
  - **SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition** [arXiv 2024] [[paper](https://arxiv.org/html/2411.15858v1)]
  - **Focus-Enhanced Scene Text Recognition with Deformable Convolutions** [arXiv 2024] [[paper](https://arxiv.org/abs/1908.10998)] [[code](https://github.com/Alpaca07/dtr)]
  - **CDistNet: Perceiving multi-domain character distance for robust text recognition** [IJCV 2024]

  ### 2023
  - **Revisiting Scene Text Recognition: A Data Perspective** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf)] [[code](https://github.com/Mountchicken/Union14M)]
  - **CLIPTER: Looking at the Bigger Picture in Scene Text Recognition** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Aberdam_CLIPTER_Looking_at_the_Bigger_Picture_in_Scene_Text_Recognition_ICCV_2023_paper.pdf)]
  - **PreSTU: Pre-Training for Scene-Text Understanding** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kil_PreSTU_Pre-Training_for_Scene-Text_Understanding_ICCV_2023_paper.pdf)]
  - **Self-Supervised Character-to-Character Distillation for Text Recognition** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guan_Self-Supervised_Character-to-Character_Distillation_for_Text_Recognition_ICCV_2023_paper.pdf)]
  - **MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition** [ICCV 2023]
  - **TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models** [AAAI 2023] [[paper](https://arxiv.org/abs/2109.10282)]
  - **CLIP4STR: A Simple Baseline for Scene Text Recognition with Pre-trained Vision-Language Model** [arXiv 2023] [[paper](https://arxiv.org/abs/2305.14014)]
  - **Relational Contrastive Learning for Scene Text Recognition** [arXiv 2023] [[paper](https://arxiv.org/abs/2308.00508)]
  - **ViTSTR-Transducer: Cross-Attention-Free Vision Transformer Transducer for Scene Text Recognition** [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10744502/)]
  - **TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition** [IJCAI 2023]

  ### 2022
  - **PARSeq: Scene Text Recognition with Permuted Autoregressive Sequence Models** [ECCV 2022] [[paper](https://arxiv.org/abs/2207.06966)] [[code](https://github.com/baudm/parseq)]
  - **Optimal Boxes: Boosting End-to-End Scene Text Recognition by Adjusting Annotated Bounding Boxes via Reinforcement Learning** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880231.pdf)]
  - **Multi-Granularity Prediction for Scene Text Recognition** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf)] 
  - **PARSeq: Scene Text Recognition with Permuted Autoregressive Sequence Models** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880301.pdf)] [[code](https://github.com/baudm/parseq)]
  - **Corner-Guided Transformer for Scene Text Recognition** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880301.pdf)]
  - **Toward Understanding WordArt: Corner-Guided Transformer for Scene Text Recognition** [ECCV 2022] [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880301.pdf)]
  - **TextAdaIN: Paying Attention to Shortcut Learning in Text Recognizers** [ECCV 2022]
  - **Multi-modal Text Recognition Networks: Interactive Enhancements between Visual and Semantic Features** [ECCV 2022]
  - **SVTR: Scene Text Recognition with a Single Visual Model** [IJCAI 2022]
  - **CarveNet: a channel-wise attention-based network for irregular scene text recognition** [IJDAR 2022] [[paper](https://link.springer.com/article/10.1007/s10032-022-00398-4)]
  - **Rethinking text rectification for scene text recognition** [Expert Systems with Applications 2023] [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423001483)]
  - **An extended attention mechanism for scene text recognition** [Expert Systems with Applications 2022] [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417422007278)]

  ### 2021
  - **ViTSTR: Vision Transformer for Fast and Efficient Scene Text Recognition** [ICDAR 2021] [[paper](https://arxiv.org/abs/2105.08582)] [[code](https://github.com/roatienza/deep-text-recognition-benchmark)]
  - **Sequence-to-Sequence Contrastive Learning for Text Recognition** [CVPR 2021] [[paper](https://www.semanticscholar.org/paper/Sequence-to-Sequence-Contrastive-Learning-for-Text-Aberdam-Litman/de9eee38b81021b3689046f72ab7c58fd7277325)]
  - **What if We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition with Fewer Labels** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf)] [[code](https://github.com/ku21fan/STR-Fewer-Labels)]
  - **Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.pdf)] [[code](https://github.com/FangShancheng/ABINet)]
   - **Sequence-to-Sequence Contrastive Learning for Text Recognition** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.pdf)]
  - **Dictionary-Guided Scene Text Recognition** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Nguyen_Dictionary-Guided_Scene_Text_Recognition_CVPR_2021_paper.html)] [[code](https://github.com/VinAIResearch/dict-guided)]
  - **Primitive Representation Learning for Scene Text Recognition** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Primitive_Representation_Learning_for_Scene_Text_Recognition_CVPR_2021_paper.pdf)] [[code](https://github.com/RuijieJ/pren)]
  - **MetaHTR: Towards Writer-Adaptive Handwritten Text Recognition** [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhunia_MetaHTR_Towards_Writer-Adaptive_Handwritten_Text_Recognition_CVPR_2021_paper.pdf)]
  - **Dictionary-Guided Scene Text Recognition** [CVPR 2021]
  - **Primitive Representation Learning for Scene Text Recognition** [CVPR 2021]
  - **MetaHTR: Towards Writer-Adaptive Handwritten Text Recognition** [CVPR 2021]
  - **ViTSTR: Vision Transformer for Fast and Efficient Scene Text Recognition** [ICDAR 2021] [[paper](https://arxiv.org/abs/2105.08582)] [[code](https://github.com/roatienza/deep-text-recognition-benchmark)]

  ### 2020
  - **SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition** [CVPR 2020] [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.pdf)] [[code](https://github.com/Pay20Y/SEED)]
  - **SCATTER: Selective Context Attentional Scene Text Recognizer** [CVPR 2020]
  - **Towards Accurate Scene Text Recognition with Semantic Reasoning Networks** [CVPR 2020] [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.pdf)]
  - **STAN: A sequential transformation attention-based network for scene text recognition** [Pattern Recognition 2021] [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304957)]
  - **GTC: Guided training of CTC** [AAAI 2020]
  - **TextScanner** [AAAI 2020]

  ### 2019
  - **A Multi-Object Rectified Attention Network for Scene Text Recognition** [Pattern Recognition] [[paper](https://arxiv.org/abs/1901.03003)]
    - https://github.com/Canjie-Luo/MORAN_v2 [PyTorch]
  - **A Simple and Robust Convolutional-Attention Network for Irregular Text Recognition** [[paper](https://arxiv.org/abs/1904.01375)]
  - **Aggregation Cross-Entropy for Sequence Recognition** [CVPR 2019] [[paper](https://arxiv.org/abs/1904.08364)]
    - https://github.com/summerlvsong/Aggregation-Cross-Entropy [PyTorch]
  - **Sequence-to-Sequence Domain Adaptation Network for Robust Text Image Recognition** [CVPR 2019] [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Sequence-To-Sequence_Domain_Adaptation_Network_for_Robust_Text_Image_Recognition_CVPR_2019_paper.pdf)]
  - **2D Attentional Irregular Scene Text Recognizer** [arXiv] [[paper](https://arxiv.org/abs/1906.05708)]
  - **Deep Neural Network for Semantic-based Text Recognition in Images** [arXiv] [[paper](https://arxiv.org/abs/1908.01403)]
  - **Symmetry-constrained Rectification Network for Scene Text Recognition** [ICCV 2019] [[paper](https://arxiv.org/abs/1908.01957)]
  - **Rethinking Irregular Scene Text Recognition (ICDAR 2019-ArT)** [[paper](https://arxiv.org/abs/1908.11834)]
    - https://github.com/Jyouhou/ICDAR2019-ArT-Recognition-Alchemy [PyTorch]
  - **Adaptive Embedding Gate for Attention-Based Scene Text Recognition** [arXiv] [[paper](https://arxiv.org/abs/1908.09475)]
  - **SAFL: A Self-Attention Scene Text Recognizer with Focal Loss** [arXiv 2022] [[paper](https://arxiv.org/pdf/2201.00132)]

  ### 2018
  - **Char-Net: A Character-Aware Neural Network for Distorted Scene Text Recognition** [AAAI 2018] [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16327/16307)]
  - **SqueezedText: A Real-time Scene Text Recognition by Binary Convolutional Encoder-decoder Network** [AAAI 2018] [[paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16354)]
  - **Edit Probability for Scene Text Recognition** [CVPR 2018] [[paper](https://arxiv.org/abs/1805.03384)]
  - **ASTER: An Attentional Scene Text Recognizer with Flexible Rectification** [TPAMI 2018] [[paper](https://ieeexplore.ieee.org/document/8395027/)]
    - https://github.com/bgshih/aster [TF]
  - **Synthetically Supervised Feature Learning for Scene Text Recognition** [ECCV 2018] [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.pdf)]
  - **Scene Text Recognition from Two-Dimensional Perspective** [AAAI 2019] [[paper](https://arxiv.org/abs/1809.06508)]
  - **ESIR: End-to-end Scene Text Recognition via Iterative Image Rectification** [CVPR 2019] [[paper](https://arxiv.org/abs/1812.05824)]

  ### 2017
  - **STN-OCR: A single Neural Network for Text Detection and Text Recognition** [arXiv] [[paper](https://arxiv.org/pdf/1707.08831.pdf)]
    - https://github.com/Bartzi/stn-ocr [MXNet]
  - **Learning to Read Irregular Text with Attention Mechanisms** [IJCAI 2017] [[paper](https://www.ijcai.org/proceedings/2017/458)]
  - **Scene Text Recognition with Sliding Convolutional Character Models** [arXiv] [[paper](https://arxiv.org/abs/1709.01727)]
  - **Focusing Attention: Towards Accurate Text Recognition in Natural Images** [ICCV 2017] [[paper](https://arxiv.org/abs/1709.02054)]
  - **AON: Towards Arbitrarily-Oriented Text Recognition** [CVPR 2018] [[paper](https://arxiv.org/abs/1711.04226)]
    - https://github.com/huizhang0110/AON [TF]
  - **Gated Recurrent Convolution Neural Network for OCR** [NIPS 2017] [[paper](https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)]
    - https://github.com/Jianfeng1991/GRCNN-for-OCR [Torch]

  ### 2016
  - **Recursive Recurrent Nets with Attention Modeling for OCR in the Wild** [CVPR 2016] [[paper](https://arxiv.org/abs/1603.03101)]
  - **Robust scene text recognition with automatic rectification** [CVPR 2016] [[paper](https://arxiv.org/abs/1603.03915)]
    - https://github.com/WarBean/tps_stn_pytorch [PyTorch]
    - https://github.com/marvis/ocr_attention [PyTorch]
  - **CNN-N-Gram for Handwriting Word Recognition** [CVPR 2016] [[paper](https://ieeexplore.ieee.org/document/7780622)]
  - **STAR-Net: A SpaTial Attention Residue Network for Scene Text Recognition** [BMVC 2016] [[paper](http://www.bmva.org/bmvc/2016/papers/paper043/paper043.pdf)]

  ### 2015
  - **Reading Scene Text in Deep Convolutional Sequences** [AAAI 2016] [[paper](https://arxiv.org/abs/1506.04395)]
  - **An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition** [TPAMI 2017] [[paper](https://arxiv.org/abs/1507.05717)]
    - https://github.com/bgshih/crnn [Torch]
    - https://github.com/weinman/cnn_lstm_ctc_ocr [TF]
    - https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow [TF]
    - https://github.com/MaybeShewill-CV/CRNN_Tensorflow [TF]
    - https://github.com/meijieru/crnn.pytorch [PyTorch]
    - https://github.com/kurapan/CRNN [Keras]

  ### 2014
  - **Deep Structured Output Learning for Unconstrained Text Recognition** [ICLR 2015] [[paper](https://arxiv.org/abs/1412.5903)]
    - https://github.com/AlexandreSev/Structured_Data [TF]
  - **Reading text in the wild with convolutional neural networks** [IJCV 2016] [[paper](https://arxiv.org/abs/1412.1842)]
    - https://github.com/mathDR/reading-text-in-the-wild [Keras]

  ---

  ## End-to-End Text Spotting

  End-to-end text spotting performs both detection and recognition in a unified framework.

  ### 2025
  - **InstructOCR: Instruction Boosting Scene Text Spotting** [arXiv 2025] [[paper](https://arxiv.org/abs/2412.15523)]
  - **dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model** [arXiv 2025] [[paper](https://arxiv.org/abs/2512.02498)]

  ### 2024
  - **FastTextSpotter: A High-Efficiency Transformer for Multilingual Scene Text Spotting** [arXiv 2024] [[paper](https://arxiv.org/abs/2408.14998)]
  - **OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition** [CVPR 2024]
  - **Bridging the Gap Between End-to-End and Two-Step Text Spotting** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Bridging_the_Gap_Between_End-to-End_and_Two-Step_Text_Spotting_CVPR_2024_paper.pdf)]
  - **InstructOCR: Instruction Boosting Scene Text Spotting** [arXiv 2024] [[paper](https://arxiv.org/html/2412.15523v1)]
  - **Arbitrary Reading Order Scene Text Spotter with Local Semantics Guidance** [arXiv 2024] [[paper](https://arxiv.org/html/2412.10159v1)]
  - **TextBlockV2: Towards Precise-Detection-Free Scene Text Spotting with Pre-trained Language Model** [arXiv 2024] [[paper](https://arxiv.org/html/2403.10047v1)]
  - **DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training** [arXiv 2024] [[paper](https://arxiv.org/html/2408.00355v1)]
  - **TransDETR: End-to-End Video Text Spotting with Transformer** [IJCV 2024] [[paper](https://link.springer.com/article/10.1007/s11263-024-02063-1)] [[code](https://github.com/weijiawu/TransDETR)]

  ### 2023
  - **DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting** [CVPR 2023] [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_DeepSolo_Let_Transformer_Decoder_With_Explicit_Points_Solo_for_Text_CVPR_2023_paper.pdf)] [[code](https://github.com/ViTAE-Transformer/DeepSolo)]
  - **ESTextSpotter: Towards Better Scene Text Spotting with Explicit Synergy in Transformer** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_ESTextSpotter_Towards_Better_Scene_Text_Spotting_with_Explicit_Synergy_in_ICCV_2023_paper.pdf)]
  - **TextFormer: A Query-based End-to-End Text Spotter with Mixed Supervision** [arXiv 2023] [[paper](https://arxiv.org/abs/2306.03377v2)]
  - **SText-DETR: End-to-End Arbitrary-Shaped Text Detection with Scalable Query in Transformer** [[paper](https://link.springer.com/chapter/10.1007/978-981-99-8546-3_39)]

  ### 2022
  - **SwinTextSpotter: Scene Text Spotting via Better Synergy Between Text Detection and Recognition** [CVPR 2022] [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_SwinTextSpotter_Scene_Text_Spotting_via_Better_Synergy_Between_Text_Detection_CVPR_2022_paper.html)] [[code](https://github.com/mxin262/SwinTextSpotter)]
  - **TESTR: Text Spotting Transformers** [arXiv 2022] [[paper](https://arxiv.org/abs/2204.01918)]
  - **TransDETR: End-to-End Video Text Spotting with Transformer** [arXiv 2022] [[paper](https://arxiv.org/abs/2203.10539)]
  - **SPTS: Single-Point Text Spotting** [ACM MM 2022]
  - **SPTS v2: Single-Point Scene Text Spotting** [TPAMI 2023]

  ### 2019
  - **Towards Unconstrained End-to-End Text Spotting** [ICCV 2019] [[paper](https://arxiv.org/abs/1908.09231)]
  - **TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting** [ICCV 2019] [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf)]
  - **Convolutional Character Networks** [ICCV 2019] [[paper](https://arxiv.org/abs/1910.07954)]

  ### 2018
  - **FOTS: Fast Oriented Text Spotting With a Unified Network** [CVPR 2018] [[paper](https://arxiv.org/abs/1801.01671)]
  - **An end-to-end TextSpotter with Explicit Alignment and Attention** [CVPR 2018] [[paper](https://arxiv.org/abs/1803.03474)]
    - https://github.com/tonghe90/textspotter [Caffe]
  - **Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes** [ECCV 2018] [[paper](https://arxiv.org/abs/1807.02242)]
    - https://github.com/lvpengyuan/masktextspotter.caffe2 [Caffe2]

  ### 2017
  - **Deep TextSpotter: An End-To-End Trainable Scene Text Localization and Recognition Framework** [ICCV 2017] [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf)]
  - **SEE: Towards Semi-Supervised End-to-End Scene Text Recognition** [AAAI 2018] [[paper](https://arxiv.org/abs/1712.05404)]
    - https://github.com/Bartzi/see [Chainer]

  ---

  ## Video Text Detection & Recognition

  ### 2024
  - **TransDETR: End-to-End Video Text Spotting with Transformer** [IJCV 2024] [[paper](https://link.springer.com/article/10.1007/s11263-024-02063-1)] [[code](https://github.com/weijiawu/TransDETR)]

  ### 2023
  - **DSText V2: A Comprehensive Video Text Spotting Dataset for Dense and Small Text** [arXiv 2023] [[paper](https://arxiv.org/abs/2312.01938)]
  - **FlowText: Synthesizing Realistic Scene Text Video with Optical Flow Estimation** [arXiv 2023] [[paper](https://arxiv.org/abs/2305.03327)]
  - **SAMText: Scalable Mask Annotation for Video Text Spotting** [arXiv 2023] [[paper](https://arxiv.org/abs/2305.01443)]
  - **ICDAR 2023 Competition on Video Text Reading for Dense and Small Text** [[paper](https://arxiv.org/abs/2304.04376)] [[competition](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_23)]

  ### 2022
  - **TransDETR: End-to-End Video Text Spotting with Transformer** [arXiv 2022] [[paper](https://arxiv.org/abs/2203.10539)]

  ### 2021
  - **BOVText: A Large-Scale, Bilingual Open World Dataset for Video Text Spotting** [arXiv 2021] [[paper](https://arxiv.org/abs/2112.04888)]
  - **ICDAR 2021 Competition on Scene Video Text Spotting** [[paper](https://arxiv.org/abs/2107.11919)]

  ### 2019
  - **You Only Recognize Once: Towards Fast Video Text Spotting** [arXiv 2019] [[paper](https://arxiv.org/abs/1903.03299)]

  ---

  ## Text Generation with Diffusion Models

  Diffusion models for rendering text in images with high quality and controllability.

  ### 2025
  - **AnyText2: Visual Text Generation and Editing With Customizable Attributes** [arXiv 2025] [[paper](https://arxiv.org/html/2411.15245v1)]
  - **TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition** [ICCV 2025] [[paper](https://arxiv.org/html/2412.01137v1)]

  ### 2024
  - **TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering** [ECCV 2024] [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72652-1_23)]

  ### 2023
  - **TextDiffuser: Diffusion Models as Text Painters** [arXiv 2023] [[paper](https://arxiv.org/abs/2305.10855)] [[project](https://jingyechen.github.io/textdiffuser/)]
  - **AnyText: Multilingual Visual Text Generation And Editing** [ICLR 2024 Spotlight] [[paper](https://arxiv.org/abs/2311.03054)] [[code](https://github.com/tyxsspa/AnyText)]

  ### Related Text Synthesis Papers

  #### 2024
  - **Text Image Inpainting via Global Structure-Guided Diffusion Models** [arXiv 2024] [[paper](https://arxiv.org/abs/2401.14832)]

  #### 2023
  - **PSGText: Stroke-Guided Scene Text Editing with PSP Module** [arXiv 2023] [[paper](https://arxiv.org/pdf/2310.13366)]
  - **Weakly supervised scene text generation for low-resource languages** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423021243)]

  #### 2019
  - **Scene Text Synthesis for Efficient and Effective Deep Network Training** [arXiv] [[paper](http://arxiv.org/abs/1901.09193)]

  #### 2016
  - **Synthetic Data for Text Localisation in Natural Images** [CVPR 2016] [[paper](https://arxiv.org/abs/1604.06646)]
    - https://github.com/ankush-me/SynthText

  ---

  ## Text Editing & Removal

  ### 2025
  - **FLUX-Text: A Simple and Advanced Diffusion Transformer Baseline for Scene Text Editing** [arXiv 2025] [[paper](https://arxiv.org/abs/2505.03329)]
  - **Recognition-Synergistic Scene Text Editing** [arXiv 2025] [[paper](https://arxiv.org/abs/2503.08387)]
  - **OmniText: A Training-Free Generalist for Controllable Text-Image Manipulation** [arXiv 2025] [[paper](https://arxiv.org/html/2510.24093)]
  - **OTR: Synthesizing Overlay Text Dataset for Text Removal** [arXiv 2025] [[paper](https://arxiv.org/html/2510.02787)]

  ### 2024
  - **DiffSTR: Controlled Diffusion Models for Scene Text Removal** [arXiv 2024] [[paper](https://arxiv.org/html/2410.21721v1)]
  - **Text Image Inpainting via Global Structure-Guided Diffusion Models** [arXiv 2024] [[paper](https://arxiv.org/html/2401.14832)]
  - **Choose What You Need: Disentangled Representation Learning for Scene Text Recognition, Removal and Editing** [CVPR 2024]

  ### 2023
  - **PSGText: Stroke-Guided Scene Text Editing with PSP Module** [arXiv 2023] [[paper](https://arxiv.org/pdf/2310.13366)]
  - **PSSTRNet: Progressive Segmentation-guided Scene Text Removal Network** [[paper](https://www.researchgate.net/publication/371536766_PSSTRNet_Progressive_Segmentation-guided_Scene_Text_Removal_Network)]

  ---

  ## Weakly Supervised Methods

  ### 2024
  - **Cps-STS: Bridging the Gap Between Content and Position for Coarse-Point-Supervised Scene Text Spotter** [IEEE TMM 2024] [[paper](https://dl.acm.org/doi/10.1109/TMM.2024.3521756)]

  ### 2023
  - **Weakly supervised scene text generation for low-resource languages** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423021243)]

  ### 2022
  - **Weakly Supervised Scene Text Detection using Deep Reinforcement Learning** [arXiv 2022] [[paper](https://arxiv.org/abs/2201.04866)]
  - **Towards Weakly-Supervised Text Spotting using a Multi-Task Transformer** [[paper](https://www.researchgate.net/publication/363906605_Towards_Weakly-Supervised_Text_Spotting_using_a_Multi-Task_Transformer)]
  - **Mixed-Supervised Scene Text Detection With Expectation-Maximization Algorithm** [IEEE 2022] [[paper](https://pubmed.ncbi.nlm.nih.gov/35976822/)]

  ### 2017
  - **Attention-Based Extraction of Structured Information from Street View Imagery** [ICDAR 2017] [[paper](https://arxiv.org/abs/1704.03549)]
  - **WeText: Scene Text Detection under Weak Supervision** [ICCV 2017] [[paper](https://arxiv.org/abs/1710.04826)]
  - **SEE: Towards Semi-Supervised End-to-End Scene Text Recognition** [AAAI 2018] [[paper](https://arxiv.org/abs/1712.05404)]
    - https://github.com/Bartzi/see [Chainer]

  ---

  ## Multilingual & Low-Resource Languages

  ### 2024
  - **Cross-Lingual Learning in Multilingual Scene Text Recognition** [ICASSP 2024] [[paper](https://arxiv.org/abs/2312.10806)] [[code](https://github.com/ku21fan/CLL-STR)]
  - **Collaborative Encoding Method for Scene Text Recognition in Low Linguistic Resources: The Uyghur Language Case Study** [Applied Sciences 2024] [[paper](https://www.mdpi.com/2076-3417/14/5/1707)]
  - **Multilingual scene text recognition: A faster R-CNN approach for Bengali and English scripts** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425040977)]
  - **An End-to-End Scene Text Recognition for Bilingual Text** [Eng 2024] [[paper](https://www.mdpi.com/2504-2289/8/9/117)]
  - **OpenOCR: Unified benchmark for Chinese and English OCR** [[info](https://huggingface.co/topdu/OpenOCR)]
  - **CDistNet: Perceiving multi-domain character distance for robust text recognition** [IJCV 2024]

  ### 2023
  - **MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition** [ICCV 2023]
  - **Chinese Text Recognition with Pre-Trained CLIP-Like Model Through Image-IDS Aligning** [arXiv 2023] [[paper](https://arxiv.org/abs/2309.01083)]
  - **TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition** [IJCAI 2023]

  ### 2019
  - **ICDAR2019 Robust Reading Challenge on Multi-lingual Scene Text Detection and Recognition (RRC-MLT-2019)**

  ---

  ## Document AI & Layout Analysis

  ### 2024
  - **LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_LayoutLLM_Layout_Instruction_Tuning_with_Large_Language_Models_for_Document_CVPR_2024_paper.pdf)]
  - **DLAFormer: An End-to-End Transformer for Document Layout Analysis** [ICDAR 2024]
  - **UNIT: Unifying Image and Text Recognition in One Vision Encoder** [NeurIPS 2024] [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/dcfb4b195fe1ec750ef13312974b620e-Paper-Conference.pdf)]
  - **Hierarchical Visual Feature Aggregation for OCR-Free Document Understanding** [NeurIPS 2024] [[poster](https://neurips.cc/virtual/2024/poster/95304)]

  ### 2023
  - **mplug-docowl 1.5: Unified structure learning for ocr-free document understanding**

  ### 2022
  - **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**
  - **DiT: Document Image Transformer**

  ### 2020-2021
  - **LayoutLM: Pre-training of Text and Layout for Document Image Understanding** [[paper](https://arxiv.org/pdf/1912.13318)]
  - **LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding**

  ---

  ## Other Scene Text Papers

  ### Mathematical Expression Recognition

  #### 2024
  - **Generating Handwritten Mathematical Expressions From Symbol Graphs: An End-to-End Pipeline** [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Generating_Handwritten_Mathematical_Expressions_From_Symbol_Graphs_An_End-to-End_Pipeline_CVPR_2024_paper.pdf)]
  - **DLAFormer: Document Layout Analysis for Formula Detection** [ICDAR 2024]
  - **Towards Scalable Training for Handwritten Mathematical Expression Recognition** [[paper](https://arxiv.org/pdf/2508.09220)]
  - **Image Over Text: Transforming Formula Recognition Evaluation with Character Detection Matching** [[paper](https://arxiv.org/html/2409.03643v2)]

  #### 2023
  - **MathWriting: A Dataset For Handwritten Mathematical Expression Recognition** [NeurIPS 2023] [[paper](https://arxiv.org/pdf/2404.10690)]
  - **ICDAR 2023 CROHME: Competition on Recognition of Handwritten Mathematical Expressions** [[paper](https://hal.science/hal-04264727/file/CROHME_2023_competition_report_paper.pdf)]
  - **Mathematical formula detection in document images: A new dataset and a new approach** [Pattern Recognition 2023] [[paper](https://dl.acm.org/doi/10.1016/j.patcog.2023.110212)]

  ### Handwritten Text Recognition

  #### 2025
  - **Handwritten Text Recognition: A Survey** [arXiv 2025] [[paper](https://arxiv.org/html/2502.08417v1)]
  - **Benchmarking Large Language Models for Handwritten Text Recognition** [[paper](https://arxiv.org/pdf/2503.15195)]
  - **GraDeT-HTR: A Resource-Efficient Bengali Handwritten Text Recognition System** [[paper](https://arxiv.org/html/2509.18081)]

  #### 2024
  - **HTR-VT: Handwritten Text Recognition with Vision Transformer** [arXiv 2024] [[paper](https://arxiv.org/abs/2409.08573)]
  - **On the Generalization of Handwritten Text Recognition Models** [[paper](https://arxiv.org/html/2411.17332v1)]
  - **Advancing Offline Handwritten Text Recognition: A Systematic Review** [[paper](https://arxiv.org/html/2507.06275v1)]

  ### OCR-Free Visual Question Answering

  #### 2024
  - **ViTextVQA: Vietnamese Text Comprehension in Images** [[paper](https://arxiv.org/html/2404.10652)]
  - **ViOCRVQA: Novel Benchmark for Vietnamese** [[paper](https://arxiv.org/html/2404.18397v1)]

  #### 2023
  - **Beyond OCR + VQA: Towards End-to-End Reading and Reasoning for Robust and Accurate TextVQA** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323000389)]

  ### Self-Supervised & Contrastive Learning

  #### 2025
  - **Self-Supervised Learning for Text Recognition: A Critical Survey** [IJCV 2025] [[paper](https://link.springer.com/article/10.1007/s11263-025-02487-3)]

  #### 2024
  - **Free Lunch: Frame-level Contrastive Learning with Text Perceiver** [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681045)]

  #### 2023
  - **Relational Contrastive Learning for Scene Text Recognition** [arXiv 2023] [[paper](https://arxiv.org/abs/2308.00508)]
  - **Self-Supervised Character-to-Character Distillation for Text Recognition** [ICCV 2023] [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guan_Self-Supervised_Character-to-Character_Distillation_for_Text_Recognition_ICCV_2023_paper.pdf)]

  #### 2021
  - **Sequence-to-Sequence Contrastive Learning for Text Recognition** [CVPR 2021] [[paper](https://www.semanticscholar.org/paper/Sequence-to-Sequence-Contrastive-Learning-for-Text-Aberdam-Litman/de9eee38b81021b3689046f72ab7c58fd7277325)]

  ### Real-Time & Mobile Deployment

  #### 2024
  - **Lumos: On-Device Scene Text Recognition for MM LLMs** [[paper](https://www.emergentmind.com/papers/2402.08017)]
  - **ACP-Net: Asymmetric Center Positioning Network for Real-Time Text Detection** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124012371)]

  #### 2023
  - **YOLOv5ST: A Lightweight and Fast Scene Text Detector** [[paper](https://www.techscience.com/cmc/v79n1/56286/html)]
  - **A light-weight natural scene text detection and recognition system** [[paper](https://link.springer.com/article/10.1007/s11042-023-15696-0)]
  - **QEST: Quantized and Efficient Scene Text Detector Using Deep Learning** [[paper](https://dl.acm.org/doi/10.1145/3526217)]

  ### Attention & Rectification Methods

  - **Attention Guided Feature Encoding for Scene Text Recognition** [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9604773/)]
  - **STAN: A sequential transformation attention-based network for scene text recognition** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304957)]
  - **Rethinking text rectification for scene text recognition** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423001483)]
  - **MTSTR: Multi-task learning for low-resolution scene text recognition via dual attention mechanism** [[paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0294943)]

  ---

  ## Contributing

  Contributions are welcome! Please feel free to submit a Pull Request. When adding papers, please:

  1. Follow the existing format
  2. Place papers in reverse chronological order
  3. Include paper links and code repositories (if available)
  4. Add `[code]` badge for papers with available implementations
  5. Use the year when the paper was first publicly available (including arXiv)

  ## License

  [![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

  To the extent possible under law, the contributors have waived all copyright and related or neighboring rights to this work.

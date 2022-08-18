[//]: # (This is the repo for reproducing the data collection process &#40;Sec. 3.1&#41; in the [PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search)

[//]: # ( paper]&#40;https://arxiv.org/abs/2207.09068&#41;.)

[//]: # ( )
[//]: # ( Stay tuned!)

[//]: # ()

<div id="top"></div>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />

<div align="center">
  <!--
  <a href="https://github.com/anguyen8/im">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->

  <h1 align="center">PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search</h1>
  <p align="center">
    Joint work between Adobe Research and Auburn University
  </p>
  
  [Thang Pham](https://twitter.com/pmthangxai), [Seunghyun Yoon](https://david-yoon.github.io), [Trung Bui](https://research.adobe.com/person/trung-bui/), and [Anh Nguyen](https://anhnguyen.me).
</div>

<!-- TABLE OF CONTENTS -->

<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#reproduce-results-for-benchmark">Reproduce results for benchmark</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->

## About The Project

Phrase in Context is a curated benchmark for phrase understanding and semantic search, consisting of three tasks of increasing difficulty: Phrase Similarity (PS), Phrase Retrieval (PR) and Phrase Sense Disambiguation (PSD). The datasets are annotated by 13 linguistic experts on Upwork and verified by two groups: ~1000 AMT crowdworkers and another set of 5 linguistic experts. PiC benchmark is distributed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

[![PiC example][pic-sample]](https://github.com/Phrase-in-Context/eval/)

:star2: Official implementation for Data Collection in our paper [PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search](https://arxiv.org/abs/2207.09068).

:sun_with_face: **Project Link**: [https://phrase-in-context.github.io/](https://phrase-in-context.github.io/)

:fire: **Online Web Demo**: https://aub.ie/phrase-search

**If you use our PiC dataset or software, please consider citing:**

    @article{pham2022PiC,
      title={PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search},
      author={Pham, Thang M and Yoon, Seunghyun and Bui, Trung and Nguyen, Anh},
      journal={arXiv preprint arXiv:2207.09068},
      year={2022}
    }

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

* Anaconda 4.10 or higher
* Python 3.9 or higher
* pip version 21 or higher

### Installation

1. Create a new folder and clone the repo

   ```sh
   mkdir phrase-in-context && cd "$_"
   git clone https://github.com/Phrase-in-Context/data-collection.git && cd data-collection
   ```

2. Create and activate a Conda environment

   ```sh
   conda create -n pic_construct python=3.9
   conda activate pic_construct
   ```

3. Install required libraries

   ```sh
   pip install -r requirements.txt
   bash extra_requirements.sh
   ```

4. Download & Prepare data
   ```sh
   bash prepare_data.sh
   ```

<!-- USAGE EXAMPLES -->

## Data Collection

As there are no English dictionaries that contain sense inventories for multi-word noun phrases (mNPs), the key **challenge** of our data collection is to find such mNPs p that (1) have multiple senses (e.g. “massive figure” means a large number but also a huge *physical shape*, depending on the context); and (2) the context documents corresponding to those senses of *p*. 

From a Wikipedia dump, we perform a 6-step procedure for mining a list of mNPs sorted descendingly by their likelihood of containing multiple senses. The most ambiguous 19,500 mNPs are then passed to experts for annotation and others for verification.

The table below is **summary** of our 3-stage data construction. **p**, *s*, *m* *d*, *q*, *l* denote target phrase, sentence, metadata, document, query, and label, respectively. In this repository, we only focus on 6-step Data Collection method to prepare data for annotation. 

[![PiC collect][pic-collect]](https://github.com/Phrase-in-Context/data-collection/)

Please check out our [paper](https://arxiv.org/abs/2207.09068) for more details and examples.

### Execute Step 1 and Step 2
1. Download Wiki articles and remove empty articles
2. Extract phrases (e.g., Noun, Proper Noun) along with their context sentences

```python
python3 extract_wiki_phrases.py
```

### Execute Step 3 to Step 6

3. Remove phrases of a single context
4. Find phrases of ambiguous words
5. Find phrases in distinct contexts
   * Sort and filter by semantic dissimilarity
   * Sort by domain dissimilarity
6. Select data for expert annotation

```python
python3 find_ambiguous_phrases.py
```


See the [open issues](https://github.com/Phrase-in-Context/eval/issues) for a full list of proposed features (and
known issues).


<!-- LICENSE -->

## License

Distributed under the MIT License.


<!-- CONTACT -->

## Contact

The entire code was done and maintained by Thang Pham, [@pmthangxai](https://twitter.com/pmthangxai) - tmp0038@auburn.edu.
Contact us via email or create github issues if you have any questions/requests. Thanks!


<!-- ACKNOWLEDGMENTS -->
<!--
## References

* Huggingface. 2022. transformers/examples/pytorch/question-answering at main · huggingface/transformers. [https://github.com/huggingface/transformers/tree/main/](https://github.com/huggingface/transformers/tree/main/) examples/pytorch/question-answering.
* Shufan Wang, Laure Thompson, and Mohit Iyyer. 2021. [Phrase-BERT](https://github.com/sf-wa-326/phrase-bert-topic-model): Improved Phrase Embeddings from BERT with an Application to Corpus Exploration. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10837–10851, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
* Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021. Learning Dense Representations of Phrases at Scale. In Association for Computational Linguistics (ACL) [DensePhrase](https://github.com/princeton-nlp/DensePhrases)
* Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. [SimCSE](https://github.com/princeton-nlp/SimCSE): Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
* Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. [Longformer](https://arxiv.org/abs/2004.05150): The long-document transformer.
-->

<p align="right">&#40;<a href="#top">back to top</a>&#41;</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Phrase-in-Context/eval.svg?style=for-the-badge
[contributors-url]: https://github.com/Phrase-in-Context/eval/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Phrase-in-Context/eval.svg?style=for-the-badge
[forks-url]: https://github.com/Phrase-in-Context/eval/network/members
[stars-shield]: https://img.shields.io/github/stars/Phrase-in-Context/eval.svg?style=for-the-badge
[stars-url]: https://github.com/Phrase-in-Context/eval/stargazers
[issues-shield]: https://img.shields.io/github/issues/Phrase-in-Context/eval.svg?style=for-the-badge
[issues-url]: https://github.com/Phrase-in-Context/eval/issues
[license-shield]: https://img.shields.io/github/license/Phrase-in-Context/eval.svg?style=for-the-badge
[license-url]: https://github.com/Phrase-in-Context/eval/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/thangpm
[pic-collect]: data_collection_summarized.png

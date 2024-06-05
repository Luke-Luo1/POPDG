---
layout: default
---

<style>
  .center {
    text-align: center;
  }
  .justify {
    text-align: justify;
  }
</style>

**People Prefer, We generate!** A brand new aesthetic-oriented music-dance dataset: PopDanceSet, combined with POPDG, let's generate the most popular dances of today!

<div class="center">
  <h1>Abstract</h1>
</div>

<div class="justify">
  Generating dances that are both lifelike and well-aligned with music continues to be a challenging task in the cross-modal domain. This paper introduces PopDanceSet, the first dataset tailored to the preferences of young audiences, enabling the generation of aesthetically oriented dances. And it surpasses the AIST++ dataset in music genre diversity and the intricacy and depth of dance movements. Moreover, the proposed POPDG model within the iDDPM framework enhances dance diversity and, through the Space Augmentation Algorithm, strengthens spatial physical connections between human body joints, ensuring that increased diversity does not compromise generation quality. A streamlined Alignment Module is also designed to improve the temporal alignment between dance and music. Extensive experiments show that POPDG achieves SOTA results on two datasets. Furthermore, the paper also expands on current evaluation metrics. The dataset and code are available at <a href="https://github.com/Luke-Luo1/POPDG">https://github.com/Luke-Luo1/POPDG</a>.
</div>

* * *

<div class="center">
  <h1>PopDanceSet</h1>
</div>

<div class="justify">
  We constructed a popularity function to filter popular dance videos on <a href="https://www.bilibili.com/v/dance/">the BiliBili website</a> (The most popular video website among young people in China), thereby creating the PopDanceSet, which features dances favored by the public. Below are some rendered examples of various dance genres from PopDanceSet:
</div>

<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/0c93f2c5-6c07-48e1-bbe8-61805beb6f6b" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/5db8a21e-62b2-49d3-9ec3-49a61b57dac1" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/15fcdf5a-8d1f-45d1-80cf-78c55d81bb34" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/760a1562-8139-4694-96dc-f8cc2dc4378e" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/c8baa595-6893-4f0d-9d0b-d9dd43d1709c" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/1223e84e-4f39-4b9f-bd91-c244fb86611b" type="video/mp4">
</video>

* * *

<div class="center">
  <h1>Pipeline</h1>
</div>

![pipeline](https://github.com/Luke-Luo1/POPDG/assets/100562982/dffc975a-a399-40eb-85e9-68527a591a86)

<div class="justify">
  The core innovative points of POPDG are threefold: First, the Space Augmentation Algorithm enhances the spatial connections between the upper body joints and the root joint in the SMPL format. Second, iDDPM maintains a balance between the quality and diversity of dance generation. Third, a streamlined alignment module is designed to improve the matching between music and dance.
</div>

* * *

<div class="center">
  <h1>Video Demo(Dedicated to Akira Toriyama, the author of Dragon Ball)</h1>
</div>

<div class="justify">
  At the beginning of March this year, Akira Toriyama (鳥山 明), the author of my favorite manga 'Dragon Ball', passed away. Therefore, I used the OP and ED songs from the Dragon Ball anime as background music and generated dances as demo displays to commemorate Akira Toriyama!
</div>

<div class="center">
  <video autoplay controls muted loop width="50%">
    <source src="https://github.com/Luke-Luo1/website/releases/download/dragonball/dragonball_1_Clip_2.mp4" type="video/mp4">
  </video>
</div>
<div class="center">
  background music: DAN DAN 心魅かれてく
</div>

<div class="center">
  <video autoplay controls muted loop width="50%">
    <source src="https://github.com/Luke-Luo1/website/releases/download/dragonball/dragonball_2_Clip_2.mp4" type="video/mp4">
  </video>
</div>
<div class="center">
  background music: ロマンティックあげるよ
</div>

<div class="center">
  <video autoplay controls muted loop width="50%">
    <source src="https://github.com/Luke-Luo1/POPDG/assets/100562982/07e1719f-9376-48cc-93d1-fd2c0303ccc8" type="video/mp4">
  </video>
</div>
<div class="center">
  background music: 魔訶不思議アドベンチャーよ
</div>

* * *

<div class="center">
  <h1>Citation</h1>
</div>

```
@article{luo2024popdg,
  title={POPDG: Popular 3D Dance Generation with PopDanceSet},
  author={Luo, Zhenye and Ren, Min and Hu, Xuecai and Huang, Yongzhen and Yao, Li},
  journal={arXiv preprint arXiv:2405.03178},
  year={2024}
}
```

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

**People Prefer, We generate!** A brand new aesthetically oriented music-dance dataset: PopDanceSet, combined with POPDG, let's generate the most popular dances of the moment!

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
  We have constructed the PopDanceSet, a database of popular dances loved by the masses, by building a popularity function to filter hot dance videos from <a href="https://www.bilibili.com/v/dance/">Bilibili</a> (China's most popular video platform among young people). Below are a few rendered examples of dance genres from the PopDanceSet:
</div>

<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/website/releases/download/dataset/1.mp4" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/website/releases/download/dataset/3.mp4" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/website/releases/download/dataset/4.mp4" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/website/releases/download/dataset/5.mp4" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/website/releases/download/dataset/6.mp4" type="video/mp4">
</video>
<video autoplay controls muted loop width="32%">
  <source src="https://github.com/Luke-Luo1/website/releases/download/dataset/7.mp4" type="video/mp4">
</video>

* * *

<div class="center">
  <h1>Pipeline</h1>
</div>

* * *

<div class="center">
  <h1>Video Demo(In memory of the author of Dragon Ball, Mr. Toriyama Akira 鳥山 明)</h1>
</div>

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```

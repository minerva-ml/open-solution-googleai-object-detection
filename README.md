# Google AI Open Images - Object Detection Track: Open Solution

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/neptune-ml/open-solution-googleai-object-detection/blob/master/LICENSE)
[![Join the chat at https://gitter.im/neptune-ml/open-solution-googleai-object-detection](https://badges.gitter.im/neptune-ml/open-solution-googleai-object-detection.svg)](https://gitter.im/neptune-ml/open-solution-googleai-object-detection?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is an open solution to the [Google AI Open Images - Object Detection Track](https://www.kaggle.com/c/google-ai-open-images-object-detection-track) :smiley:

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository :computer:](https://github.com/neptune-ml/open-solution-googleai-object-detection). This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [Google-AI-Object-Detection-Challenge :chart_with_upwards_trend:](https://app.neptune.ml/neptune-ml/Google-AI-Object-Detection-Challenge) and images below:

| UNet training monitor :bar_chart: | Predicted bounding boxes :bar_chart: |
|:---|:---|
|[![unet-training-monitor](https://gist.githubusercontent.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/19272701575bca235473adaabb7b7c54b2416a54/gai-1.png)](https://app.neptune.ml/-/dashboard/experiment/f945da64-6dd3-459b-94c5-58bc6a83f590)|[![predicted-bounding-boxes](https://gist.githubusercontent.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/19272701575bca235473adaabb7b7c54b2416a54/gai-2.png)](https://app.neptune.ml/-/dashboard/experiment/c779468e-d3f7-44b8-a3a4-43a012315708)|

## Dataset for this competition
This competition is special, because it used [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/index.html), which is quite large: `>1.8M` images and `>0.5TB` :astonished: To make it more approachable, we are hosting entire dataset in the neptune's public directory :sunglasses:. **You can use this dataset in [neptune.ml](https://neptune.ml) with no additional setup :+1:.**

## Learn more about our solutions
[Kaggle discussion](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion) is our primary way of communication, however, we are also documenting our work on the [Wiki pages :blue_book:](https://github.com/neptune-ml/open-solution-googleai-object-detection/wiki). Click on the dolphin to get started [:dolphin:](https://github.com/neptune-ml/open-solution-googleai-object-detection/wiki).

## Disclaimer
In this open source solution you will find references to the [neptune.ml](https://neptune.ml). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Installation
### Fast Track
1. Clone repository, install `tensorflow 1.6`, `PyTorch 0.3.1` and then remaining requirements (check _requirements.txt_)

```bash
pip3 install -r requirements.txt
```

2. Register to the [neptune.ml](https://neptune.ml/login) _(if you wish to use it)_
3. Run experiment:

:trident:
```bash
neptune run will appear here soon :)
```

:snake:
```bash
python command will appear here soon :)
```

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/neptune-ml/open-solution-googleai-object-detection/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. [Kaggle discussion](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion) is our primary way of communication.
1. Read project's [Wiki](https://github.com/neptune-ml/open-solution-googleai-object-detection/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](https://neptune.ml).
1. Submit an [issue]((https://github.com/neptune-ml/open-solution-googleai-object-detection/issues)) directly in this repo.

# audio-analysis-dash

This is the source code for the Genre Classification App, this contains the entire code (along with a few missteps and trials in the test.py file. As of this commit, the main file still has debugging code, that will provide information on memory consumption on running. It can be easily removed once downloaded, however. I will also be cleaning up the code and file structure shortly.

To use, download/clone the repo and run `python3 application.py`, this will perform all of the other required tasks.

**Other pertinent information:**
- the current code runs a tflite model, however, I have also shared the entire tensorflow keras SavedModel, and that can be used to provide appropriate predictions as well.

- the data three genres specified - top, second, third. This is because the presentation of these genres and their interpretation is a little more involved than would seem at first glance. Please visit [the live app](http://bit.ly/genre-classification-app) to learn more, as I'll be adding information to explain my reasoning either on there, or in this README file at a later date.


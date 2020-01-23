# Analyzing Book Covers for Suggestions

Is a picture worth a thousand words? Book covers are the first glimpse into the book and need to immediately convey a convincing message to induce readers. But how do you design a book cover and what should you prioritize? We use Amazon product and review data to explore book covers. Using nearest neighbor and convolutional neural network models, we aim to learn what is a good book cover and help design novel covers by generating suggestions.

Often, conventional cover design requires hiring expensive art designers over a prolonged undefined and non-standardized process. Designers often have their own style derived by a personal artistic system, all of which is designed to entice potential readers to open a new book. However, if the goal is to have more readers, or be a more successful book, why not explore whether there is a relationship between cover design and success directly?

We use image feature extraction and machine learning techniques to learn successful covers for books. By doing so, we discover certain elements of book covers are quite related to the eventual success of a book as defined by Amazon reviews and overall rating compared to similar covers. We can judge a book by its cover, and we can generate suggestions developed to enhance the success of the book. Importantly, we also show what successful books are similar to the proposed cover to give the user an idea of how to improve their cover stylistically.

[![Project Video](https://img.youtube.com/vi/g_RFR9QapU4/0.jpg)](https://youtu.be/g_RFR9QapU4)

Dataset: Mainly, we will rely on the Amazon Review Data with respect to books. This contains 51,311,621 reviews for 2,935,525 books containing product information, links to images, and metadata. Related to this data is a curated set of 207,572 cleaned book cover images of size 224x224 that may be easier to rely upon as some product images are more than just the cover. Both are available online for free. Our software can support more cover images of size 224x224 with metadata including title, ASIN id, reviews, and overall rating.

Language: Python, HTML, JavaScript, Pandas, NumPy, Keras, SKLearn, Google Dataproc, Flask, Docker on Google Cloud Build, Kubernetes, Google Cloud Storage, Google BigQuery

Analytics: We analyzed each book review and consolidated ratings to an overall rating for all books in our data set. We then extract image features such as color vibrancy from each book cover and generate to a k-nearest neighbors model. We then trained a convolutional neural network on cover images to determine whether there is a learnable relationship between book cover and successfulness by overall rating. Given the strong relationship, we combined the k-nearest neighbor and CNN models to build a web app suggestion engine that, given a potential book cover, can give you an estimated rating and some recommendations to improve.

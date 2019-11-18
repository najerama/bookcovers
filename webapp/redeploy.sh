set -e

NEW_TAG=gcr.io/eecs-e6893-book-cover/flaskapp:`date +%s`

gcloud builds --project eecs-e6893-book-cover submit --tag $NEW_TAG .

kubectl set image deployment/flask-app-tutorial flask-app-tutorial=$NEW_TAG

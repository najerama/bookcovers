Followed guide here to setup this module:
https://medium.com/google-cloud/a-guide-to-deploy-flask-app-on-google-kubernetes-engine-bfbbee5c6fb

Make whatever changes you want to app.py then redeploy with

./redeploy.sh

This script does assume some background configuration (your gcloud and kubeconfigs are pointing to the right place, for example). If you run into issues, ping me (charlie) and I'll help you debug.

This now requires the bigquery credentials to deploy. Ping Charlie about it if necessary!

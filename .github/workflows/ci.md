name: Deploy MLFLOW using github

on:
  workflow_dispatch:
    inputs:
      model_stage:
        description: "Choose model stage"
        required: true
        default: "staging"
        type: choice
        options:
          - staging
          - production
          - archive
  
jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: test
        run: echo "workflow working"
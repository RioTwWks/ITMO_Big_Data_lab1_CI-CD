pipeline {
  agent any
  options {
    buildDiscarder(logRotator(numToKeepStr: '5'))
  }
  environment {
    DOCKERHUB_CREDENTIALS = credentials('riotwwks-dockerhub')
  }
  stages {
    stage('Build') {
      steps {
        sh 'docker build -t riotwwks/rt-test:latest .'
      }
    }
    stage('Login') {
      steps {
        sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
      }
    }
    stage('Push') {
      steps {
        sh 'docker push riotwwks/rt-test:latest'
      }
    }
  }
  post {
    always {
      sh 'docker logout'
    }
  }
}

FROM node:20-alpine

WORKDIR /tool
RUN npm init -y >/dev/null 2>&1
RUN npm install --no-audit --no-fund ajv@8.17.1

WORKDIR /app

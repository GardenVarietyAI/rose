FROM golang:1.22-alpine AS build
WORKDIR /app

RUN apk add --no-cache ca-certificates

COPY main.go ./main.go
RUN CGO_ENABLED=0 GOOS=linux go build -o aproxy main.go

FROM alpine:3.20
WORKDIR /app
RUN apk add --no-cache ca-certificates curl

COPY --from=build /app/aproxy /app/aproxy

EXPOSE 8000
ENTRYPOINT ["/app/aproxy"]

use anyhow::Result;
use clap::Parser;
use tokio_tungstenite::{accept_hdr_async, tungstenite::Message};
use tracing::{error, info, warn};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};

mod models;
mod server;
mod error;

use server::InferenceServer;

#[derive(Parser, Debug)]
#[command(name = "rose-inference-rs")]
#[command(about = "High-performance Rust inference server for Rose")]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:8005")]
    bind: SocketAddr,

    #[arg(short, long, default_value = "cpu")]
    device: String,

    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_max_level(args.log_level.parse().unwrap_or(tracing::Level::INFO))
        .init();

    info!("Starting Rose Inference Server on {}", args.bind);

    let server = InferenceServer::new(args.device).await?;
    let listener = TcpListener::bind(&args.bind).await?;

    info!("WebSocket server listening on {} with /inference endpoint", args.bind);

    let shutdown = tokio::signal::ctrl_c();

    tokio::select! {
        _ = shutdown => {
            info!("Received shutdown signal, gracefully shutting down...");
        }
        _ = accept_connections(listener, server) => {
            info!("Server stopped accepting connections");
        }
    }

    Ok(())
}

async fn accept_connections(listener: TcpListener, server: InferenceServer) -> Result<()> {
    while let Ok((stream, addr)) = listener.accept().await {
        info!("New connection from {}", addr);
        let server_clone = server.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, server_clone).await {
                error!("Connection error: {}", e);
            }
        });
    }
    Ok(())
}

async fn handle_connection(stream: TcpStream, server: InferenceServer) -> Result<()> {
    let callback = |req: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
        if req.uri().path() != "/inference" {
            warn!("Invalid path: {}, expected /inference", req.uri().path());
            return Err(tokio_tungstenite::tungstenite::handshake::server::ErrorResponse::new(Some("404".to_string())));
        }
        Ok(response)
    };

    let ws_stream = accept_hdr_async(stream, callback).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    while let Some(msg) = ws_receiver.next().await {
        match msg? {
            Message::Text(text) => {
                match server.process_streaming_request(&text, &mut ws_sender).await {
                    Ok(()) => {
                        // Streaming completed successfully
                    }
                    Err(e) => {
                        warn!("Request processing error: {}", e);
                        let error_response = serde_json::to_string(&server::InferenceResponse::Error {
                            error: e.to_string(),
                        })?;
                        if let Err(e) = ws_sender.send(Message::Text(error_response)).await {
                            error!("Failed to send error response: {}", e);
                            break;
                        }
                    }
                }
            }
            Message::Close(_) => {
                info!("Client disconnected");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

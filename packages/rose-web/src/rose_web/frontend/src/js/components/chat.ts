import { fetchEventSource } from '@microsoft/fetch-event-source';
import { el, mount, setChildren } from 'redom';
import type { ChatKitEvent, ChatMessage } from '../types.js';
import './messages/assistant-message.js';
import './messages/system-message.js';
import './messages/user-message.js';

class RoseChat extends HTMLElement {
  private messages: ChatMessage[] = [];
  private threadId: string | null = null;
  private isLoading: boolean = false;
  private model!: string;
  private messagesContainer!: HTMLElement;
  private input!: HTMLInputElement;
  private sendButton!: HTMLButtonElement;
  private currentAssistantMessage: ChatMessage | null = null;

  connectedCallback() {
    this.model = this.dataset.model || 'Qwen--Qwen3-1.7B-GGUF';
    this.render();
  }

  render() {
    this.messagesContainer = el('div', {
      id: 'chat-messages',
      className: 'flex-1 overflow-y-auto p-4 flex flex-col gap-3'
    });

    this.input = el('input', {
      type: 'text',
      placeholder: 'Type your message...',
      className: 'flex-1 p-3 bg-neutral-900 text-neutral-100 border border-neutral-700 rounded-lg text-base focus:outline-none focus:border-rose-500 disabled:opacity-50',
      onkeyup: (e: KeyboardEvent) => {
        if (e.key === 'Enter') this.sendMessage();
      }
    }) as HTMLInputElement;

    this.sendButton = el('button', {
      className: 'px-5 py-3 bg-rose-500 text-white border-none rounded-lg font-semibold cursor-pointer transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed',
      onclick: () => this.sendMessage()
    }, 'Send') as HTMLButtonElement;

    const container = el('div', { className: 'w-full h-full flex flex-col relative' },
      this.messagesContainer,
      el('div', { className: 'flex gap-2 p-4 bg-neutral-800 border-t border-neutral-700' },
        this.input,
        this.sendButton
      )
    );

    mount(this, container);
  }

  addMessage(message: ChatMessage) {
    this.messages.push(message);
    this.updateMessages();
  }

  updateMessages() {
    const messageElements = this.messages.map(msg => this.createMessageElement(msg));

    if (this.isLoading) {
      messageElements.push(this.createLoadingElement());
    }

    setChildren(this.messagesContainer, messageElements);
    this.scrollToBottom();
  }

  createMessageElement(message: ChatMessage) {
    const elementName = `${message.role}-message`;
    const element = document.createElement(elementName) as any;
    element.content = message.content;
    return element;
  }

  createLoadingElement() {
    return el('div', { className: 'self-start bg-transparent p-0' },
      el('div', { className: 'flex gap-1' },
        el('span', { className: 'w-2 h-2 bg-neutral-500 rounded-full animate-bounce', style: 'animation-delay: -0.32s' }),
        el('span', { className: 'w-2 h-2 bg-neutral-500 rounded-full animate-bounce', style: 'animation-delay: -0.16s' }),
        el('span', { className: 'w-2 h-2 bg-neutral-500 rounded-full animate-bounce' })
      )
    );
  }

  async sendMessage() {
    const userMessage = this.input.value.trim();
    if (!userMessage || this.isLoading) return;

    this.input.value = '';
    this.input.focus();
    this.addMessage({ role: 'user', content: userMessage });
    this.isLoading = true;
    this.updateLoadingState();

    try {
      await this.streamChatKitResponse(userMessage);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.addMessage({ role: 'system', content: `Error: ${errorMessage}` });
    } finally {
      this.isLoading = false;
      this.updateLoadingState();
    }
  }


  async streamChatKitResponse(message: string) {
    const requestType = this.threadId ? 'threads.add_user_message' : 'threads.create';
    const requestBody: any = {
      type: requestType,
      params: {
        input: {
          content: [{ type: 'input_text', text: message }],
          attachments: [],
          quoted_text: null,
          inference_options: { model: this.model }
        }
      }
    };

    if (this.threadId) {
      requestBody.params.thread_id = this.threadId;
    }

    await fetchEventSource('/chatkit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      onopen: async (response) => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      },
      onmessage: (event) => {
        const parsed = JSON.parse(event.data);
        this.handleChatKitEvent(parsed);
      },
      onerror: (err) => {
        console.error('SSE error:', err);
        throw err;
      }
    });
  }

  handleChatKitEvent(event: ChatKitEvent) {
    switch (event.type) {
      case 'thread.created':
        if (event.thread) {
          this.threadId = event.thread.id;
        }
        break;

      case 'thread.item.updated':
        if (event.update?.type === 'assistant_message.content_part.text_delta') {
          if (!this.currentAssistantMessage || this.currentAssistantMessage.itemId !== event.item_id) {
            this.isLoading = false;
            this.currentAssistantMessage = {
              role: 'assistant',
              content: event.update.delta || '',
              itemId: event.item_id
            };
            this.addMessage(this.currentAssistantMessage);
          } else {
            this.currentAssistantMessage.content = (this.currentAssistantMessage.content || '') + (event.update.delta || '');
            this.updateMessages();
          }
        }
        break;

      case 'thread.item.done':
        this.currentAssistantMessage = null;
        break;
    }
  }

  updateLoadingState() {
    this.input.disabled = this.isLoading;
    this.sendButton.disabled = this.isLoading;
    this.updateMessages();
  }

  scrollToBottom() {
    requestAnimationFrame(() => {
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    });
  }

}

customElements.define('rose-chat', RoseChat);

export default RoseChat;

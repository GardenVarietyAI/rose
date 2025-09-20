import { el, setAttr, setChildren } from 'redom';

class AssistantMessage extends HTMLElement {
  content?: string;

  connectedCallback() {
    setAttr(this, { class: 'self-start bg-neutral-800 text-neutral-100 p-3 rounded-lg max-w-[80%]' });
    setChildren(this, [
      el('div', { className: 'whitespace-pre-wrap break-words' }, this.content || '')
    ]);
  }
}

customElements.define('assistant-message', AssistantMessage);
export default AssistantMessage;

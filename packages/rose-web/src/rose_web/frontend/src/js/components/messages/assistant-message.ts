import { el, setAttr, setChildren } from 'redom';

class AssistantMessage extends HTMLElement {
  content?: string;

  connectedCallback() {
    setAttr(this, { class: 'self-start text-neutral-100 max-w-[80%]' });
    setChildren(this, [
      el('div', { className: 'whitespace-pre-wrap break-words' }, this.content || '')
    ]);
  }
}

customElements.define('assistant-message', AssistantMessage);
export default AssistantMessage;

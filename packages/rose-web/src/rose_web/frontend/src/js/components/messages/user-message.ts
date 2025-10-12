import { el, setAttr, setChildren } from 'redom';

class UserMessage extends HTMLElement {
  content?: string;

  connectedCallback() {
    setAttr(this, { class: 'self-end bg-rose-500 text-white p-3 rounded-lg max-w-[80%]' });
    setChildren(this, [
      el('div', { className: 'whitespace-pre-wrap break-words' }, this.content || '')
    ]);
  }
}

customElements.define('user-message', UserMessage);
export default UserMessage;

import { setAttr, setChildren, text } from 'redom';

class SystemMessage extends HTMLElement {
  content?: string;

  connectedCallback() {
    setAttr(this, { class: 'self-center bg-rose-900 text-rose-100 p-2 rounded text-sm max-w-full' });
    setChildren(this, [text(this.content || '')]);
  }
}

customElements.define('system-message', SystemMessage);
export default SystemMessage;

import { el, mount } from 'redom';

export class FineTuningJobComponent extends HTMLElement {
  connectedCallback() {
    this.render();
  }

  render() {
    const view = el('div.view-container',
      el('h2', 'Fine-tuning Jobs'),
      el('p.muted', 'Fine-tuning interface coming soon...'),
      this.buildStats()
    );

    mount(this, view);
  }

  buildStats() {
    return el('div.stats-box',
      el('h3', 'Quick Stats'),
      el('p', 'Models fine-tuned: 0'),
      el('p', 'Training jobs: 0'),
      el('p', 'Datasets: 0')
    );
  }
}

customElements.define('fine-tuning-job', FineTuningJobComponent);

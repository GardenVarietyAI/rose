export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content?: string;
  itemId?: string;
}

export interface ChatKitEvent {
  type: string;
  thread?: { id: string };
  item_id?: string;
  item?: {
    type: string;
    id?: string;
  };
  update?: {
    type: string;
    delta?: string;
    content_index?: number;
  };
}

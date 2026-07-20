import axios from 'axios';

interface PredictionResult {
  intent: string;
  confidence: number;
  target: {
    type: string | null;
    attribute: string | null;
    label: string | null;
    index: string | null;
    relation: string | null;
    reference: string | null;
  };
}

class InferenceService {
  private baseUrl: string = 'http://localhost:5000/api';
  private isLocalModel: boolean = true;

  INTENTS = [
    'click', 'tap', 'double_tap', 'long_press', 'scroll', 'swipe',
    'drag', 'drop', 'delete', 'copy', 'paste', 'highlight', 'select',
    'open', 'close', 'play', 'pause', 'stop', 'search', 'zoom',
    'rotate', 'move', 'resize', 'upload', 'download', 'share', 'save',
    'refresh', 'retry', 'enable', 'disable', 'check', 'uncheck',
    'expand', 'collapse', 'accept', 'reject', 'approve', 'back',
    'next', 'previous', 'home', 'settings', 'login', 'logout',
    'bookmark', 'pin', 'unpin', 'start', 'finish', 'exit', 'continue',
    'add', 'remove', 'hide', 'show', 'mute', 'unmute', 'install',
    'uninstall', 'submit', 'cancel', 'type', 'enter', 'focus',
    'hover', 'launch'
  ];

  TARGETS = [
    'button', 'icon', 'image', 'checkbox', 'switch', 'radio_button',
    'dropdown', 'textbox', 'input', 'text', 'chart', 'graph', 'table',
    'card', 'menu', 'navigation_bar', 'toolbar', 'search_box', 'video',
    'audio', 'attachment', 'file', 'document', 'message', 'notification',
    'popup', 'dialog', 'tab', 'list_item', 'profile', 'avatar',
    'calendar', 'slider', 'progress_bar', 'link', 'qr_code', 'camera',
    'gallery', 'email', 'password', 'phone_number'
  ];

  SPATIAL_RELATIONS = [
    'top', 'bottom', 'left', 'right', 'top_left', 'top_right',
    'bottom_left', 'bottom_right', 'center', 'center_left', 'center_right',
    'upper_center', 'lower_center', 'above', 'below', 'beside',
    'adjacent_to', 'next_to', 'inside', 'outside', 'between',
    'before', 'after', 'behind', 'in_front_of', 'beneath', 'over',
    'under', 'nearest', 'farthest', 'first', 'second', 'third',
    'last', 'second_last', 'first_row', 'second_row', 'first_column',
    'last_column', 'top_edge', 'bottom_edge', 'left_edge', 'right_edge'
  ];

  async predict(command: string): Promise<PredictionResult> {
    try {
      if (this.isLocalModel) {
        return await this.predictLocal(command);
      } else {
        return await this.predictRemote(command);
      }
    } catch (error) {
      console.error('Inference error:', error);
      return this.createFallbackPrediction(command);
    }
  }

  private async predictLocal(command: string): Promise<PredictionResult> {
    const intentId = this.classifyIntent(command);
    const intent = this.INTENTS[intentId] || 'click';
    const confidence = this.calculateConfidence(command, intent);
    const target = this.extractTargetInfo(command);

    return {
      intent,
      confidence,
      target,
    };
  }

  private async predictRemote(command: string): Promise<PredictionResult> {
    try {
      const response = await axios.post(`${this.baseUrl}/predict`, {
        command,
      }, {
        timeout: 5000,
      });
      return response.data;
    } catch (error) {
      console.warn('Remote inference failed, falling back to local:', error);
      this.isLocalModel = true;
      return this.predictLocal(command);
    }
  }

  private classifyIntent(command: string): number {
    const commandLower = command.toLowerCase();

    const intentKeywords: { [key: string]: string[] } = {
      'click': ['click', 'tap', 'press'],
      'scroll': ['scroll', 'swipe', 'drag'],
      'delete': ['delete', 'remove', 'clear'],
      'search': ['search', 'find', 'look'],
      'open': ['open', 'launch', 'start'],
      'close': ['close', 'exit', 'back'],
      'submit': ['submit', 'send', 'confirm'],
      'cancel': ['cancel', 'abort', 'escape'],
      'copy': ['copy', 'duplicate'],
      'paste': ['paste', 'paste in'],
      'save': ['save', 'store'],
      'download': ['download', 'get'],
      'upload': ['upload', 'send'],
      'share': ['share', 'distribute'],
      'refresh': ['refresh', 'reload'],
      'zoom': ['zoom', 'magnify'],
      'rotate': ['rotate', 'turn'],
      'select': ['select', 'choose', 'pick'],
      'add': ['add', 'create', 'new'],
      'enable': ['enable', 'turn on'],
      'disable': ['disable', 'turn off'],
    };

    for (const [intent, keywords] of Object.entries(intentKeywords)) {
      for (const keyword of keywords) {
        if (commandLower.includes(keyword)) {
          const intentIndex = this.INTENTS.indexOf(intent);
          return intentIndex >= 0 ? intentIndex : 0;
        }
      }
    }

    return 0;
  }

  private calculateConfidence(command: string, intent: string): number {
    let confidence = 0.85;

    if (command.length < 5) confidence -= 0.1;
    if (command.length > 100) confidence -= 0.05;

    const intentKeywords: { [key: string]: string[] } = {
      'click': ['click', 'tap'],
      'scroll': ['scroll'],
      'delete': ['delete', 'remove'],
    };

    const keywords = intentKeywords[intent] || [];
    if (keywords.some(k => command.toLowerCase().includes(k))) {
      confidence += 0.1;
    }

    return Math.min(0.99, Math.max(0.5, confidence));
  }

  private extractTargetInfo(command: string): PredictionResult['target'] {
    const commandLower = command.toLowerCase();
    const target: PredictionResult['target'] = {
      type: null,
      attribute: null,
      label: null,
      index: null,
      relation: null,
      reference: null,
    };

    for (const t of this.TARGETS) {
      if (commandLower.includes(t.replace('_', ' '))) {
        target.type = t;
        break;
      }
    }

    const spatialKeywords: { [key: string]: string } = {
      'top': 'top',
      'bottom': 'bottom',
      'left': 'left',
      'right': 'right',
      'first': 'first',
      'second': 'second',
      'last': 'last',
      'above': 'above',
      'below': 'below',
      'next': 'next',
      'previous': 'previous',
    };

    for (const [keyword, relation] of Object.entries(spatialKeywords)) {
      if (commandLower.includes(keyword)) {
        target.relation = relation;
        break;
      }
    }

    const attributes = ['red', 'blue', 'green', 'large', 'small', 'primary', 'secondary'];
    for (const attr of attributes) {
      if (commandLower.includes(attr)) {
        target.attribute = attr;
        break;
      }
    }

    return target;
  }

  private createFallbackPrediction(command: string): PredictionResult {
    return {
      intent: 'click',
      confidence: 0.5,
      target: {
        type: null,
        attribute: null,
        label: null,
        index: null,
        relation: null,
        reference: null,
      },
    };
  }
}

export default new InferenceService();

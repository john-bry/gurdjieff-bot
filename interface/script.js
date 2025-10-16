class GurdjieffBot {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.isConnected = false;
        this.currentSources = [];
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkConnection();
    }
    
    initializeElements() {
        this.$statusText = $('#status-text');
        this.$docCount = $('#doc-count');
        this.$chatMessages = $('#chat-messages');
        this.$messageInput = $('#message-input');
        this.$sendButton = $('#send-button');
        this.$sourcesPanel = $('#sources-panel');
        this.$sourcesContent = $('#sources-content');
        this.$closeSources = $('#close-sources');
    }
    
    attachEventListeners() {
        this.$sendButton.on('click', () => this.sendMessage());
        this.$messageInput.on('keypress', (e) => {
            if (e.which === 13) this.sendMessage();
        });
        this.$closeSources.on('click', () => this.hideSources());
        
        // Close sources panel when clicking outside
        $(document).on('click', (e) => {
            if (!$(e.target).closest('.sources-panel, .sources-indicator').length) {
                this.hideSources();
            }
        });
    }
    
    async checkConnection() {
        try {
            this.$statusText.text('Connecting...').removeClass().addClass('connecting');
            
            const response = await $.get(`${this.apiUrl}/health`);
            
            if (response.api === 'healthy') {
                this.isConnected = true;
                this.$statusText.text('Connected').removeClass().addClass('connected');
                this.$docCount.text(response.document_count || 0);
                
                if (response.openai_key !== 'configured') {
                    this.addBotMessage('⚠️ OpenAI API key is not configured. Please set up your environment variables.');
                }
                
                if (response.document_count === 0) {
                    this.addBotMessage('ℹ️ No documents loaded yet. Please run the data processing pipeline first.');
                }
                
                this.enableChat();
            } else {
                throw new Error('API unhealthy');
            }
        } catch (error) {
            this.isConnected = false;
            this.$statusText.text('Disconnected').removeClass().addClass('disconnected');
            this.$docCount.text('-');
            this.disableChat();
            this.addBotMessage('❌ Unable to connect to the server. Please make sure the server is running.');
            console.error('Connection error:', error);
        }
    }
    
    enableChat() {
        this.$messageInput.prop('disabled', false);
        this.$sendButton.prop('disabled', false);
        this.$messageInput.focus();
    }
    
    disableChat() {
        this.$messageInput.prop('disabled', true);
        this.$sendButton.prop('disabled', true);
    }
    
    async sendMessage() {
        const message = this.$messageInput.val().trim();
        if (!message || !this.isConnected) return;
        
        // Add user message
        this.addUserMessage(message);
        this.$messageInput.val('');
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await $.ajax({
                url: `${this.apiUrl}/chat`,
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ message: message }),
                timeout: 30000
            });
            
            this.hideTypingIndicator();
            this.addBotMessage(response.response, response.sources);
            
        } catch (error) {
            this.hideTypingIndicator();
            console.error('Chat error:', error);
            
            let errorMessage = '❌ Sorry, I encountered an error. ';
            
            if (error.status === 0) {
                errorMessage += 'Please check if the server is running.';
            } else if (error.status === 500) {
                errorMessage += 'Server error. Please try again.';
            } else if (error.statusText === 'timeout') {
                errorMessage += 'Request timed out. Please try again.';
            } else {
                errorMessage += 'Please try again.';
            }
            
            this.addBotMessage(errorMessage);
        }
    }
    
    addUserMessage(message) {
        const messageHtml = `
            <div class="message user-message">
                <div class="message-content">
                    ${this.escapeHtml(message)}
                </div>
            </div>
        `;
        this.$chatMessages.append(messageHtml);
        this.scrollToBottom();
    }
    
    addBotMessage(message, sources = []) {
        this.currentSources = sources || [];
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="sources-indicator" onclick="window.gurdjieffBot.showSources()">
                    📚 ${sources.length} source(s) - Click to view
                </div>
            `;
        }
        
        const messageHtml = `
            <div class="message bot-message">
                <div class="message-content">
                    ${this.formatMessage(message)}
                    ${sourcesHtml}
                </div>
            </div>
        `;
        this.$chatMessages.append(messageHtml);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const typingHtml = `
            <div class="message bot-message typing-message">
                <div class="message-content">
                    <div class="typing-indicator">
                        <span>Thinking</span>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.$chatMessages.append(typingHtml);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        $('.typing-message').remove();
    }
    
    showSources() {
        if (this.currentSources.length === 0) return;
        
        let sourcesHtml = '';
        this.currentSources.forEach((source, index) => {
            sourcesHtml += `
                <div class="source-item">
                    <div class="source-meta">
                        <strong>Source ${index + 1}:</strong> ${source.source} 
                        (Chunk ${source.chunk_id}, Similarity: ${(1 - source.distance).toFixed(3)})
                    </div>
                    <div class="source-text">${this.escapeHtml(source.preview)}</div>
                </div>
            `;
        });
        
        this.$sourcesContent.html(sourcesHtml);
        this.$sourcesPanel.removeClass('hidden');
    }
    
    hideSources() {
        this.$sourcesPanel.addClass('hidden');
    }
    
    formatMessage(message) {
        // Convert newlines to <br> and preserve basic formatting
        return this.escapeHtml(message)
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>')
            .replace(/<p><\/p>/g, '');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    scrollToBottom() {
        this.$chatMessages.scrollTop(this.$chatMessages[0].scrollHeight);
    }
}

// Initialize the bot when document is ready
$(document).ready(() => {
    window.gurdjieffBot = new GurdjieffBot();
});
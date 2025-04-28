$(document).ready(function() {
    let waitingForRecommendation = false;
	let wait_confirmation = false;

    // Function to scroll chat to bottom
    function scrollToBottom() {
        var chatContainer = document.getElementById('chat-container');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Scroll to bottom on page load
    scrollToBottom();

    // Handle form submission
    $('#chat-form').submit(function(e) {
        e.preventDefault();

        var message = $('#message').val();
        if (message.trim() === '') return;

        // Get current timestamp
        const now = new Date();
        const timestamp = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

        // Add user message to chat
        $('#chat-container').append(`
            <div class="flex items-end justify-end space-x-2 message-container">
                <div class="bg-white p-3 rounded-lg border-2 border-black">
                    <p class="text-right font-bold">${userName}</p>
                    <p>${message}</p>
                    <div class="timestamp text-right">${timestamp}</div>
                </div>
                <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
            </div>
        `);

        $('#message').val('');
        scrollToBottom();

        // Show typing indicator
        $('#chat-container').append(`
            <div class="flex items-start space-x-2" id="typing-indicator">
                <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
                <div class="bg-white p-3 rounded-lg border-2 border-black">
                    <p class="font-bold">ChatBot</p>
                    <p>Mengetik<span class="typing-dots">...</span></p>
                </div>
            </div>
        `);
        scrollToBottom();
        
        console.log("waitingForRecommendation status:", waitingForRecommendation);
		console.log("waitingForConfirmation status:", wait_confirmation);		// debug log

        // Percabangan untuk input rekomendasi
        if (waitingForRecommendation) {
            $.ajax({
                url: `${baseUrl}${activeController}/sendbook`,
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ message: message }),
                dataType: 'json',
                xhrFields: {
                    withCredentials: true
                },
                success: function(response) {
                    $('#typing-indicator').remove();
                    
                    // Tambahkan respons dari server yang sudah diformat dengan timestamp
                    $('#chat-container').append(`
                        <div class="flex items-start space-x-2 message-container">
                            <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
                            <div class="bg-white p-3 rounded-lg border-2 border-black">
                                <p class="font-bold">ChatBot</p>
                                <div>${response.response}</div>
                                <div class="timestamp">${timestamp}</div>
                            </div>
                        </div>
                    `);
                    
                    // Reset state ke mode intent normal
                    waitingForRecommendation = false;
					wait_confirmation= true;
                    scrollToBottom();
                },
                error: function(xhr, status, error) {
					console.error('XHR:', xhr);
					console.error('Status:', status);
					console.error('Error:', error);
				
					$('#typing-indicator').remove();
					let serverResponse = '';
					try {
						serverResponse = JSON.parse(xhr.responseText).response || "Terjadi kesalahan saat merekomendasikan buku. Silakan coba lagi.";
					} catch (e) {
						serverResponse = "Terjadi kesalahan saat merekomendasikan buku. Silakan coba lagi.";
					}

					$('#chat-container').append(`
						<div class="flex items-start space-x-2 message-container">
							<div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
							<div class="bg-white p-3 rounded-lg border-2 border-black">
								<p class="font-bold">ChatBot</p>
								<div>${serverResponse}</div>
								<div class="timestamp">${timestamp}</div>
							</div>
						</div>
					`);
					waitingForRecommendation = false;
					wait_confirmation = false;
					
					scrollToBottom();
				}

            });
            return; // keluar dari fungsi karena kita sudah tangani
        }

        // Jika bukan rekomendasi, lanjut ke intent biasa
        $.ajax({
            url: `${baseUrl}${activeController}/send`,
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message, wait_confirmation: wait_confirmation }),
            dataType: 'json',
            xhrFields: {
                withCredentials: true
            },
            success: function(response) {
				$('#typing-indicator').remove();
				$('#chat-container').append(`
					<div class="flex items-start space-x-2 message-container">
						<div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
						<div class="bg-white p-3 rounded-lg border-2 border-black">
							<p class="font-bold">ChatBot</p>
							<div>${response.response}</div>
							<div class="timestamp">${timestamp}</div>
						</div>
					</div>
				`);
				scrollToBottom();

				if (response.next_action && response.next_action === 'wait_book_recommendation') {
					waitingForRecommendation = true;
					wait_confirmation = false;   // entering recommendation
				} else if (response.next_action && response.next_action === 'confirmation') {
					waitingForRecommendation = true;
					// wait_confirmation stays true
				} else {
					// No special next_action
					waitingForRecommendation = false;
					wait_confirmation = false;   // reset after everything
				}
			},
            error: function(xhr, status, error) {
                $('#typing-indicator').remove();
                $('#chat-container').append(`
                    <div class="flex items-start space-x-2 message-container">
                        <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
                        <div class="bg-white p-3 rounded-lg border-2 border-black">
                            <p class="font-bold">ChatBot</p>
                            <p>Maaf, terjadi kesalahan. Silakan coba lagi.</p>
                            <div class="timestamp">${timestamp}</div>
                        </div>
                    </div>
                `);
                scrollToBottom();
            }
        });
    });

    // Animate typing dots
    setInterval(function() {
        var dots = $('.typing-dots');
        if (dots.length > 0) {
            var text = dots.text();
            dots.text(text.length >= 3 ? '' : text + '.');
        }
    }, 500);
});
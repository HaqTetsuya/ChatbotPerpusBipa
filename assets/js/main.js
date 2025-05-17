$(document).ready(function () {
    let waitingForRecommendation = false;
    let wait_confirmation = false;

    function scrollToBottom() {
        var chatContainer = document.getElementById('chat-container');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    scrollToBottom();

    // Handle chat suggestion clicks
    $('.chat-suggestion').on('click', function() {
        const suggestionText = $(this).find('p').text();
        $('#message').val(suggestionText);
        $('#chat-form').submit();
    });

    $('#chat-form').submit(function (e) {
        e.preventDefault();

        // Prevent user input while bot is typing
        if ($('#typing-indicator').length > 0) {
            console.log("Bot masih mengetik, kirim pesan diblokir.");
            return;
        }

        var message = $('#message').val();
        if (message.trim() === '') return;

        // Hide suggestion bubbles when user sends a message
        $('.chat-suggestion').slideUp(300, function() {
            $(this).remove();
        });

        const now = new Date();
        const timestamp = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        $('#chat-container').append(`
            <div class="flex items-end justify-end space-x-3 message-container">
                <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                    <p class="text-right font-bold handwriting text-lg">${userName}</p>
                    <p>${message}</p>
                    <div class="timestamp text-right">${timestamp}</div>
                </div>
                <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                    <i class="fas fa-user"></i>
                </div>
            </div>
        `);

        $('#message').val('');
        scrollToBottom();

        $('#chat-container').append(`
            <div class="flex items-start space-x-3" id="typing-indicator">
                <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                    <i class="fas fa-book"></i>
                </div>
                <div class="bg-white p-4 rounded-lg border-2 border-black max-w-[80%]">
                    <p class="font-bold handwriting text-lg">Perpus Bina Patria</p>
                    <p>Mengetik<span class="typing-dots">...</span></p>
                </div>
            </div>
        `);
        scrollToBottom();

        console.log("waitingForRecommendation:", waitingForRecommendation);
        console.log("wait_confirmation:", wait_confirmation);

        if (waitingForRecommendation) {
            $.ajax({
                url: `${baseUrl}${activeController}/sendbook`,
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ message: message }),
                dataType: 'json',
                xhrFields: { withCredentials: true },
                success: function (response) {
                    $('#typing-indicator').remove();

                    let responseHtml = `
                        <div class="flex items-start space-x-3 message-container">
                            <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                                <i class="fas fa-book"></i>
                            </div>
                            <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                                <p class="font-bold handwriting text-lg">Perpus Bina Patria</p>
                                <div>${response.response}</div>
                                <div class="timestamp">${timestamp}</div>
                            </div>
                        </div>`;

                    // Tambahkan tombol "Lihat lebih banyak" jika low_recommendation
                    if (response.low_recommendation) {
                        responseHtml += `
                            <div class="flex justify-start mb-3 ml-14 mt-2">
                                <button id="more-recommendation-btn" class="handwriting bg-white px-4 py-2 border-2 border-black rounded-lg shadow-sm hover:bg-gray-50 transition-all">
                                    Lihat lebih banyak rekomendasi
                                </button>
                            </div>`;
                    }

                    $('#chat-container').append(responseHtml);

                    waitingForRecommendation = false;
                    wait_confirmation = true;
                    scrollToBottom();
                },
                error: function (xhr) {
                    $('#typing-indicator').remove();

                    let serverResponse = "Terjadi kesalahan saat merekomendasikan buku. Silakan coba lagi.";
                    try {
                        serverResponse = JSON.parse(xhr.responseText).response || serverResponse;
                    } catch (e) { }

                    $('#chat-container').append(`
                        <div class="flex items-start space-x-3 message-container">
                            <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                                <i class="fas fa-book"></i>
                            </div>
                            <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                                <p class="font-bold handwriting text-lg">Perpus Bina Patria</p>
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
            return;
        }

        // Normal intent
        $.ajax({
            url: `${baseUrl}${activeController}/send`,
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message, wait_confirmation: wait_confirmation }),
            dataType: 'json',
            xhrFields: { withCredentials: true },
            success: function (response) {
                $('#typing-indicator').remove();

                let responseHtml = `
                    <div class="flex items-start space-x-3 message-container">
                        <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                            <i class="fas fa-book"></i>
                        </div>
                        <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                            <p class="font-bold handwriting text-lg">Perpus Bina Patria</p>
                            <div>${response.response}</div>
                            <div class="timestamp">${timestamp}</div>
                        </div>
                    </div>`;

                // Tambah tombol "Lihat lebih banyak rekomendasi" jika low_recommendation
                if (response.low_recommendation) {
                    responseHtml += `
                        <div class="flex justify-start mb-3 ml-14 mt-2">
                            <button id="more-recommendation-btn" class="handwriting bg-white px-4 py-2 border-2 border-black rounded-lg shadow-sm hover:bg-gray-50 transition-all">
                                Lihat lebih banyak rekomendasi
                            </button>
                        </div>`;
                }

                $('#chat-container').append(responseHtml);
                scrollToBottom();

                // Handle next action
                if (response.next_action === 'wait_book_recommendation') {
                    waitingForRecommendation = true;
                    wait_confirmation = false;
                } else if (response.next_action === 'confirmation') {
                    waitingForRecommendation = true;
                    wait_confirmation = true;
                } else {
                    waitingForRecommendation = false;
                    wait_confirmation = false;
                }
            },
            error: function () {
                $('#typing-indicator').remove();
                $('#chat-container').append(`
                    <div class="flex items-start space-x-3 message-container">
                        <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                            <i class="fas fa-book"></i>
                        </div>
                        <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                            <p class="font-bold handwriting text-lg">Perpus Bina Patria</p>
                            <p>Maaf, terjadi kesalahan. Silakan coba lagi.</p>
                            <div class="timestamp">${timestamp}</div>
                        </div>
                    </div>
                `);
                scrollToBottom();
            }
        });
    });

    // Event delegation for handling "Lihat lebih banyak rekomendasi" button
    $(document).on('click', '#more-recommendation-btn', function () {
        const message = "Lanjutkan rekomendasi buku";
        $('#message').val(message);
        $('#chat-form').submit();
    });

    setInterval(function () {
        var dots = $('.typing-dots');
        if (dots.length > 0) {
            var text = dots.text();
            dots.text(text.length >= 3 ? '' : text + '.');
        }
    }, 500);
});
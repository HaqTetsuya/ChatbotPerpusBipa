$(document).ready(function () {
	let waitingForRecommendation = false;
	let wait_confirmation = false;

	function scrollToBottom() {
		var chatContainer = document.getElementById('chat-container');
		chatContainer.scrollTop = chatContainer.scrollHeight;
	}

	scrollToBottom();

	$('#chat-form').submit(function (e) {
		e.preventDefault();

		// Prevent user input while bot is typing
		if ($('#typing-indicator').length > 0) {
			console.log("Bot masih mengetik, kirim pesan diblokir.");
			return;
		}

		var message = $('#message').val();
		if (message.trim() === '') return;

		const now = new Date();
		const timestamp = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

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
						<div class="flex items-start space-x-2 message-container">
							<div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
							<div class="bg-white p-3 rounded-lg border-2 border-black">
								<p class="font-bold">ChatBot</p>
								<div>${response.response}</div>
								<div class="timestamp">${timestamp}</div>
							</div>
						</div>`;

					// Tambahkan tombol "Lihat lebih banyak" jika low_recommendation
					if (response.low_recommendation) {
						responseHtml += `
							<div class="flex justify-start mb-2 ml-12">
								<button id="more-recommendation-btn" class="text-blue-600 underline">
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
			return;
			$(document).on('click', '#more-recommendation-btn', function () {
				const message = "Lanjutkan rekomendasi buku"; // atau teks khusus
				$('#message').val(message);
				$('#chat-form').submit();
			});

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
					<div class="flex items-start space-x-2 message-container">
						<div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0"></div>
						<div class="bg-white p-3 rounded-lg border-2 border-black">
							<p class="font-bold">ChatBot</p>
							<div>${response.response}</div>
							<div class="timestamp">${timestamp}</div>
						</div>
					</div>`;

				// Tambah tombol "Lihat lebih banyak rekomendasi" jika low_recommendation
				if (response.low_recommendation && waitingForRecommendation) {
					responseHtml += `
						<div class="flex justify-start mb-2 ml-12">
							<button id="more-recommendation-btn" class="text-blue-600 underline">
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

	setInterval(function () {
		var dots = $('.typing-dots');
		if (dots.length > 0) {
			var text = dots.text();
			dots.text(text.length >= 3 ? '' : text + '.');
		}
	}, 500);
});

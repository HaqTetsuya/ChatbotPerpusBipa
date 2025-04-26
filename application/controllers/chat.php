<?php
defined('BASEPATH') or exit('No direct script access allowed');
class chat extends CI_Controller
{
    public function __construct()
    {
        parent::__construct();
		$this->load->helper('responses');
        $this->load->model('Chat_model', 'chatModel'); // Memuat model Chat_model
		$this->load->model('m_account');
		if (!$this->session->userdata('status') || $this->session->userdata('status') !== 'telah_login') {
			redirect('auth/login?alert=belum_login');
		}
    }

    public function index()
    {
		$user_id= $this->session->userdata('id');
    
    // Ambil data pengguna dari tabel users
		$user = $this->m_account->getUserById($user_id);        
		$data = [
			'active_controller' => 'chat',
			'chats' => $this->chatModel->getChatHistoryByUser('chats', $user_id), // Gunakan tabel 'chats2',
			'user' => $user
		];		
        $this->load->view('chatForm', $data);
    }

    public function send()
    {
        if (!$this->input->is_ajax_request()) {
            echo json_encode(['response' => 'Invalid request']);
            return;
        }
        $user_id= $this->session->userdata('id');
		log_message('error', 'AJAX - Session user_id: ' . print_r($user_id, true));
        $data = json_decode(file_get_contents('php://input'), true);
        $message = $data['message'] ?? '';

        // Panggil API Flask untuk menganalisis intent
        $ch = curl_init('http://localhost:5000/api/analyze');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode(["text" => $message]));
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
        $response = curl_exec($ch);
        $httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        error_log("Raw Flask Response: " . $response); // Log Flask response
        
        $result = [
            'response' => '',
            'next_action' => null
        ];

        if ($httpcode == 200) {
            $responseData = json_decode($response, true);
            
            if (!isset($responseData['intent'])) {
                error_log("Flask API did not return an intent.");
                $result['response'] = 'Terjadi kesalahan saat memproses intent.';
            } else {
                $intent = strtolower($responseData['intent']); // Ensure lowercase matching
                $confidence = $responseData['confidence'] ?? 0;
                error_log("Intent received: $intent, Confidence: $confidence");
				$bot_output = get_bot_response($intent);

				$response_text = $bot_output['response'];
				$next_action = isset($bot_output['next_action']) ? $bot_output['next_action'] : null;

				$result['response'] = $response_text;
				if ($next_action) {
					$result['next_action'] = $next_action;
				}
				// Simpan percakapan ke database           
				$data = [
					'user' => $user_id,
					'user_message' => $message,
					'bot_response' => $result['response']
				];
				$this->db->db_debug = TRUE;
				$this->chatModel->saveChat('chats', $data);

            }

        } else {
            error_log("Flask API error: HTTP $httpcode");
            $result['response'] = 'Terjadi kesalahan saat menghubungi server. Silakan coba lagi.';
        }
        
        echo json_encode($result);
    }

	
    public function sendbook() {
		if (!$this->input->is_ajax_request()) {
			show_error('Direct access not allowed', 403);
			return;
		}

		$user_id = $this->session->userdata('id');
		$json_data = file_get_contents('php://input');
		$post_data = json_decode($json_data, true);
		$message = isset($post_data['message']) ? trim($post_data['message']) : '';

		if (empty($message)) {
			$this->output
				->set_content_type('application/json')
				->set_output(json_encode(['error' => 'No message provided']));
			return;
		}

		$api_data = [
			'query' => $message,
			'top_n' => 5
		];

		$ch = curl_init('http://localhost:5000/api/recommend'); // perhatikan endpoint-nya
		curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
		curl_setopt($ch, CURLOPT_POST, true);
		curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($api_data));
		curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);

		$response = curl_exec($ch);
		$status_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
		$curl_error = curl_error($ch);
		curl_close($ch);

		if ($response === false) {
			log_message('error', 'cURL Error: ' . $curl_error);
			$this->output
				->set_content_type('application/json')
				->set_output(json_encode(['error' => 'Failed to connect to recommendation service', 'details' => $curl_error]));
			return;
		}

		$api_response = json_decode($response, true);

		if ($response === false || $status_code >= 500) {
			// Jika tidak bisa konek sama sekali atau error server
			$error = isset($api_response['error']) ? $api_response['error'] : 'Unknown error';
			log_message('error', 'API Error: ' . $error . ' (Status: ' . $status_code . ')');
			$this->output
				->set_content_type('application/json')
				->set_status_header(500)
				->set_output(json_encode(['error' => 'Recommendation service error', 'details' => $error]));
			return;
		}

		// Kalau sukses walaupun recommendations kosong, tetap lanjut
		$recommendations = $api_response['recommendations'] ?? [];

		$formatted_response = $this->format_recommendations($recommendations);

		$data = [
			'user' => $user_id,
			'user_message' => $message,
			'bot_response' => $formatted_response
		];

		$this->chatModel->saveChat('chats', $data);

		$this->output->set_content_type('application/json')->set_output(json_encode(['response' => $formatted_response]));

	}

    /**
     * Format book recommendations into HTML for display
     */
    private function format_recommendations($results) {
        $output = "<strong>Buku yang direkomendasikan untuk Anda:</strong><br><br>";
        
        foreach ($results as $index => $book) {
            $relevance_percentage = $book['relevance'] * 100;
            $year = $book['year'] ? $book['year'] : 'Tahun tidak diketahui';
            
            $output .= "<div class='book-recommendation'>";
            $output .= "<strong>" . ($index + 1) . ". " . htmlspecialchars($book['title']) . "</strong><br>";
            $output .= "Penulis: " . htmlspecialchars($book['author']) . "<br>";
            $output .= "Kategori: " . htmlspecialchars($book['category']) . "<br>";
            $output .= "Tahun: " . htmlspecialchars($year) . "<br>";
            $output .= "<p><em>Deskripsi:</em> " . nl2br(htmlspecialchars($book['description'])) . "</p>";
            $output .= "Relevansi: " . number_format($relevance_percentage, 0) . "%<br>";
            $output .= "</div><br>";
        }
        
        $output .= "<p>Apakah Anda ingin rekomendasi buku lainnya? Silakan ketik pertanyaan atau topik yang Anda minati.</p>";
        
        return $output;
    }
    
    public function clear()
    {
        $this->chatModel->clearChatHistory('chats');
        redirect('chat'); // redirect back to chat page
    }
}
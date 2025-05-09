<?php
defined('BASEPATH') or exit('No script direct access allowed');
class Auth extends CI_Controller
{
    function __construct()
    {
        parent::__construct();
        $this->load->model('m_account');
    }

    function login()
    {
        $this->load->view('loginForm');
    }
    public function aksiLogin()
    {
        // Validate form fields
        $this->form_validation->set_rules('email', 'Email', 'required|valid_email');
        $this->form_validation->set_rules('password', 'Password', 'required');

        if ($this->form_validation->run() !== false) {
            $email = $this->input->post('email', TRUE); // XSS filtering
            $password = $this->input->post('password', TRUE);

            // Query user from DB by email only
            $user = $this->m_account->get_user_by_email($email);

            if ($user) {
                // Check password (assumes it's hashed using password_hash())
                if ($password == $user->password) {
                    // Create session
                    $data_session = array(
                        'id' => $user->id,
                        'email' => $user->email,
                        'status' => 'telah_login'
                    );
                    $this->session->set_userdata($data_session);

                    redirect('welcome');
                } else {
                    // Wrong password
                    redirect('auth/login?alert=password_salah');
                }
            } else {
                // Email not found
                redirect('auth/login?alert=email_tidak_terdaftar');
            }
        } else {
            // Validation failed
            redirect('auth/login?alert=validasi_gagal');
        }
    }

    function signup()
    {
        $this->load->view('signupForm');
    }
	public function aksiSign()
	{
		// Validation Rules
		$this->form_validation->set_rules('nama', 'Nama', 'required');
		$this->form_validation->set_rules('email', 'Email', 'required|valid_email');
		$this->form_validation->set_rules('password', 'Password', 'required|min_length[8]');
		$this->form_validation->set_rules('confirm_password', 'Konfirmasi Password', 'required|matches[password]');

		// If validation fails
		if ($this->form_validation->run() === false) {
			$this->session->set_flashdata('errors', $this->form_validation->error_array());
			redirect('auth/signup');
			return;
		}

		// Clean user input (with XSS filtering)
		$nama     = $this->input->post('nama', TRUE);
		$email    = $this->input->post('email', TRUE);
		$password = $this->input->post('password', TRUE);

		// Check if email is already registered
		$cek = $this->m_account->cek_email_exist($email);
		if ($cek->num_rows() > 0) {
			redirect('auth/signup?alert=duplikat');
			return;
		}

		// Prepare data to insert
		$data = array(
			'nama'     => $nama,
			'email'    => $email,
			'password' => $password // as requested: no hashing logic included
		);

		// Insert user to DB
		$insert = $this->m_account->insert_user($data);
		if (!$insert) {
			redirect('auth/signup?alert=gagal_insert');
			return;
		}

		// Get the inserted user ID
		$user_id = $this->db->insert_id();

		// Create session
		$data_session = array(
			'id'     => $user_id,
			'email'  => $email,
			'status' => 'telah_login'
		);
		$this->session->set_userdata($data_session);

		// Redirect to welcome page
		redirect('welcome');
	}

	
	function logout(){
        $this->session->sess_destroy();
        $url = base_url('');
        redirect('auth/login');
	}
}

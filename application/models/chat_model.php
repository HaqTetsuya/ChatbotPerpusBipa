<?php
class Chat_model extends CI_Model
{

    public function saveChat($table, $data)
    {
        $this->db->insert($table, $data);
    }

    public function getChatHistory($table)
    {
        $this->db->order_by('timestamp', 'ASC');
        return $this->db->get($table)->result_array(); // Ambil chat history dari tabel yang sesuai
    }

    public function clearChatHistory($table, $userId)
	{
		return $this->db->where('user', $userId)->delete($table);
	}
	
    public function getChatHistoryByUser($table, $user_id)
    {
        $this->db->where('user', $user_id);
        $this->db->order_by('timestamp', 'ASC'); // jika ada kolom waktu
        return $this->db->get($table)->result_array(); // bukan ->result()

    }

}

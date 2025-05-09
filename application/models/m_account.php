<?php
class M_account extends CI_Model
{
    function cek_login($table, $where)
    {
        return $this->db->get_where($table, $where);
    }
    function update_data($table, $data, $where)
    {
        $this->db->where($where);
        $this->db->update($table, $data);
    }
    function get_data($table)
    {
        return $this->db->get($table);
    }
    public function insert_user($data)
	{
		return $this->db->insert('users', $data); // Replace 'users' with your actual table name
	}

    function edit_data($table, $where)
    {
        return $this->db->get_where($table, $where);
    }
    function delete_data($table, $where)
    {
        return $this->db->delete($table, $where);
    }
    public function cek_email_exist($email)
    {
        return $this->db->get_where('users', ['email' => $email]);
    }
    public function get_user_by_email($email)
	{
		return $this->db->get_where('users', ['email' => $email])->row();
	}
	public function getUserById($id)
	{
		return $this->db->get_where('users', ['id' => $id])->row();
	}


}

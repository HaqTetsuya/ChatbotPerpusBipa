<?php

defined('BASEPATH') OR exit('No direct script access allowed');

if (!function_exists('get_bot_response')) {
    function get_bot_response($intent, $data = null) {
        switch ($intent) {
            case 'greeting':
                return [
                    'response' => '<p><strong>Halo!</strong> Ada yang bisa saya bantu hari ini? 😊</p>'
                ];

            case 'goodbye':
                return [
                    'response' => '<p><strong>Sampai jumpa!</strong> Semoga harimu menyenangkan! 👋</p>'
                ];

            case 'jam_layanan':
                return [
                    'response' => '
                        <strong>Jam Layanan Perpustakaan:</strong><br>
                        Pelayanan menggunakan sistem tertutup (Closed Access) dengan jam layanan sebagai berikut:<br><br>
                        <ul>
                            <li><strong>Senin s.d Kamis</strong>: 08.00 - 15.30</li>
                            <li><strong>Jumat & Sabtu</strong>: 08.00 - 18.00</li>
                            <li><strong>Minggu</strong>: 08.00 - 12.00</li>
                        </ul>'
                ];

            case 'keanggotaan':
                return [
                    'response' => '
                        <strong>Informasi Keanggotaan:</strong><br>
                        Untuk menjadi anggota perpustakaan, Anda harus:<br>
                        <ol>
                            <li>Mengisi formulir pendaftaran</li>
                            <li>Melampirkan fotokopi identitas (KTP/KTM)</li>
                            <li>Menyerahkan pas foto ukuran 3x4 sebanyak 2 lembar</li>
                        </ol>
                        <p>Setelah diverifikasi, Anda akan mendapatkan kartu anggota untuk meminjam buku.</p>'
                ];

            case 'cari_buku':
                return [
                    'response' => '
                        <p><strong>Pencarian Buku:</strong><br>
                        Silakan ketikkan judul atau pengarang buku yang Anda cari.<br>
                        <em>Contoh:</em> cari buku pemrograman Python</p>',
                    'next_action' => 'wait_book_recommendation'
                ];

            case 'confirm':
                return [
                    'response' => '<p><strong>Baik, terima kasih!</strong> Permintaan Anda telah dikonfirmasi. ✅</p>'
                ];

            case 'denied':
                return [
                    'response' => '<p><strong>Permintaan dibatalkan.</strong> Jika butuh bantuan lain, silakan sampaikan. ❌</p>'
                ];
				
			case 'unknown':
                return [
                    'response' => '<p>Maaf, saya tidak mengerti perintah tersebut.</p>'
                ];

            default:
                return [
                    'response' => '<p>Maaf, saya belum bisa memproses permintaan tersebut.</p>'
                ];
        }
    }
}

<?php

defined('BASEPATH') OR exit('No direct script access allowed');

if (!function_exists('get_bot_response')) {
    function get_bot_response($intent, $data = null) {
        switch ($intent) {
            case 'greeting':
                return [
                    'response' => '<p><strong>Halo, Sir!</strong> ?? Selamat datang di Perpustakaan Digital. Ada yang bisa saya bantu hari ini? ??</p>'
                ];

            case 'goodbye':
                return [
                    'response' => '<p><strong>Sampai jumpa, Sir!</strong> Semoga harimu penuh pengetahuan dan inspirasi. ??</p>'
                ];

            case 'jam_layanan':
                return [
                    'response' =>
                        '<p><strong>?? Jam Layanan Perpustakaan:</strong></p>' .
                        '<ul>' .
                        '<li><strong>Senin - Kamis</strong>: 08.00 - 15.30</li>' .
                        '<li><strong>Jumat & Sabtu</strong>: 08.00 - 18.00</li>' .
                        '<li><strong>Minggu</strong>: 08.00 - 12.00</li>' .
                        '</ul>'
                ];

            case 'keanggotaan':
                return [
                    'response' =>
                        '<p><strong>?? Informasi Keanggotaan:</strong></p>' .
                        '<ol>' .
                        '<li>Mengisi formulir pendaftaran online/di tempat</li>' .
                        '<li>Melampirkan fotokopi KTP/KTM</li>' .
                        '<li>Menyerahkan 2 lembar pas foto 3x4</li>' .
                        '</ol>' .
                        '<p>Setelah data diverifikasi, kartu anggota akan dikirimkan via email/diambil langsung. Kartu ini memungkinkan Anda meminjam buku fisik dan mengakses koleksi digital.</p>'.
						'<p>Bagi mahasiswa dari luar STMIK BINA PATRIA MAGELANG diperbolehkan memanfaatkan fasilitas perpustakaan, kecuali peminjaman buku. Syarat untuk memanfaatkan fasilitas adalah dengan menunjukkan kartu mahasiswa yang masih berlaku.</p>'			
                ];
			
			case 'fasilitas':
				return [
					'response' => 
						'<p><strong> Daftar Fasilitas:</p></strong>'.
						'<ol>' .
						'<li>Peminjaman buku teks, 2 eksemplar selama 7 hari kerja, dengan denda keterlambatan 200 per hari per eksemplar</li>'.
						'<li>Baca di tempat</li>'.
						'<li>Penelusuran literatur</li>'.
						'<li>Fotokopi koleksi</li>'.
						'<li>Ruang baca</li>'.
						'<li>Internet Gratis</li>'.
						'<li>Locker tas/jaket</li>'.
						'</ol>' .
						'<p>Bagi mahasiswa dari luar STMIK BINA PATRIA MAGELANG diperbolehkan memanfaatkan fasilitas perpustakaan, kecuali peminjaman buku. Syarat untuk memanfaatkan fasilitas adalah dengan menunjukkan kartu mahasiswa yang masih berlaku.</p>'			
				];

            case 'cari_buku':
                return [
                    'response' =>
                        '<p><strong>?? Pencarian Buku:</strong><br>' .
                        'Ketik judul, pengarang, atau kata kunci buku yang Anda cari.<br>' .
                        '<em>Contoh:</em> cari buku "Pemrograman Python untuk Pemula"</p>',
                    'next_action' => 'wait_book_recommendation'
                ];


            case 'confirm':
                $waitConfirmation = $data['wait_confirmation'] ?? false;
                if ($waitConfirmation) {
                    return [
                        'response' =>
                            '<p><strong>?? Pencarian Buku:</strong><br>' .
                            'Silakan masukkan kembali judul atau deskripsi buku Anda.</p>',
                        'next_action' => 'confirmation'
                    ];
                } else {
                    return [
                        'response' => '<p><strong>?? Terima kasih, Sir!</strong> Permintaan Anda sudah kami proses.</p>'
                    ];
                }

            case 'denied':
                return [
                    'response' => '<p><strong>? Permintaan dibatalkan, Sir.</strong> Silakan sampaikan jika ada yang lain.</p>'
                ];

            case 'cara_pinjam':
                return [
                    'response' =>
                        '<p><strong>?? Prosedur Peminjaman & Pengembalian Buku:</strong></p>' .
                        '<p><em>Cara Peminjaman:</em></p>' .
                        '<ol>' .
                        '<li>Cari buku yang diinginkan melalui sistem <em>search</em> atau katalog.</li>' .
                        '<li>Pastikan buku tersedia (status: Tersedia).</li>' .
                        '<li>Datang ke meja sirkulasi dengan membawa kartu anggota.</li>' .
                        '<li>Serahkan kartu anggota dan beri tahu petugas judul buku.</li>' .
                        '<li>Buku akan diproses, kemudian Anda menerima slip peminjaman beserta tanggal kembali.</li>' .
                        '</ol>' .
                        '<p><em>Aturan Peminjaman:</em></p>' .
                        '<ul>' .
                        '<li>Maksimum 3 buku per anggota.</li>' .
                        '<li>Durasi peminjaman: 7 hari kerja.</li>' .
                        '<li>Perpanjangan bisa dilakukan 1 kali jika belum ada pemesan.</li>' .
                        '</ul>' .
                        '<p><em>Prosedur Pengembalian:</em></p>' .
                        '<ol>' .
                        '<li>Kembalikan buku ke meja sirkulasi atau <em>drop box</em> setelah jam layanan.</li>' .
                        '<li>Petugas akan memeriksa kondisi buku dan menghapus catatan peminjaman.</li>' .
                        '<li>Jika terlambat, denda Rp1.000/hari/buku akan dikenakan.</li>' .
                        '</ol>' .
                        '<p>Semoga membantu, Sir! ??</p>'
                ];

            case 'unknown':
                return [
                    'response' => '<p>Maaf, Sir, saya belum memahami maksud Anda. Bisa dijelaskan lagi? ??</p>'
                ];

            default:
                return [
                    'response' => '<p>Maaf, Sir, saya belum bisa memproses permintaan tersebut.</p>'
                ];
        }
    }
}

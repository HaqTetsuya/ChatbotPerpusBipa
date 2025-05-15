<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Interface</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&family=Crimson+Text:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link href="<?php echo base_url(); ?>assets/css/main.css" rel="stylesheet">
    
    <style>
        body {
            font-family: 'Crimson Text', serif;
            background-color: #f5f5f0;
            background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1h98v98H1V1zm1 1v96h96V2H2z' fill='%23e0e0e0' fill-opacity='0.4' fill-rule='evenodd'/%3E%3C/svg%3E");
        }
        
        .chat-wrapper {
            box-shadow: 8px 8px 0 rgba(0,0,0,0.2);
            position: relative;
        }
        
        /* Page curl effect */
        .chat-wrapper:after {
            content: "";
            position: absolute;
            bottom: 0;
            right: 0;
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, transparent 50%, #e0e0e0 50%);
            border-radius: 0 0 5px 0;
        }
        
        .message-bubble {
            transition: all 0.2s ease;
        }
        
        .message-bubble:hover {
            transform: translateY(-2px);
            box-shadow: 3px 3px 0 rgba(0,0,0,0.1);
        }
        
        .timestamp {
            font-family: 'Caveat', cursive;
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.5rem;
        }
        
        #message::placeholder {
            font-style: italic;
            color: #aaa;
        }
        
        .handwriting {
            font-family: 'Caveat', cursive;
        }
        
        .chat-suggestion {
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .chat-suggestion:hover {
            transform: translateY(-3px);
            box-shadow: 3px 3px 0 rgba(0,0,0,0.2);
        }
    </style>
</head>
<body class="h-screen flex items-center justify-center p-4">
    <div class="chat-wrapper bg-white w-full max-w-4xl rounded-xl shadow-lg p-6 flex flex-col h-full border-2 border-black">
        <!-- Header with profile and controls -->
        <div class="flex justify-between items-center mb-6 pb-3 border-b-2 border-gray-200">
            <div class="flex items-center">
                <div class="w-10 h-10 border-2 border-black rounded-full mr-3 flex items-center justify-center">
                    <i class="fas fa-book-open"></i>
                </div>
                <h1 class="text-2xl font-bold handwriting">BookChat</h1>
            </div>
            
            <div class="flex items-center space-x-4">                               
                <!-- Profile dropdown -->
                <div class="profile-dropdown">
                    <button class="bg-white border-2 border-black rounded-full p-2 hover:bg-gray-100 transition-colors">
                        <i class="fas fa-user"></i>
                    </button>
                    <div class="dropdown-content rounded-lg">
                        <div class="px-4 py-3 border-b border-gray-200">
                            <p class="font-bold handwriting text-lg"><?= $user->nama; ?></p>
                            <p class="text-sm"><?= $user->email; ?></p>
                            <p class="text-xs text-gray-500">ID: <?= $user->id; ?></p>
                        </div>
						<a href="<?php echo site_url($active_controller.'/clear'); ?>" onclick="return confirm('Yakin ingin menghapus semua riwayat chat Anda?')" class="flex items-center px-4 py-2 hover:bg-gray-100">
							<i class="fas fa-trash mr-2"></i> Hapus Riwayat
						</a>
                        <a href="<?php echo site_url('auth/logout'); ?>" class="flex items-center px-4 py-2 hover:bg-gray-100 rounded-b-lg">
                            <i class="fas fa-sign-out-alt mr-2"></i> Logout
                        </a>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Chat container -->
        <div class="flex flex-col space-y-4 flex-grow overflow-y-auto bg-white/80 p-4 rounded-lg" id="chat-container">
            <!-- Welcome message -->
            <div class="flex items-start space-x-3 message-container">
                <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                    <i class="fas fa-book"></i>
                </div>
                <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                    <p class="font-bold handwriting text-lg">BookChat</p>
                    <p>Halo <?= $user->nama; ?>, Ada yang bisa saya bantu dengan rekomendasikan buku?</p>
                    <div class="timestamp">Hari ini, <?php echo date('H:i'); ?></div>
                </div>
            </div>
            
            <?php if(empty($chats)): ?>
            <!-- Chat suggestions only shown when no chat history -->
            <div class="flex flex-col space-y-3 mt-4 ml-12">
                <div class="chat-suggestion bg-white/90 p-3 rounded-lg border-2 border-black max-w-[80%] hover:bg-gray-50">
                    <p class="handwriting">halo selamat pagi</p>
                </div>
                
                <div class="chat-suggestion bg-white/90 p-3 rounded-lg border-2 border-black max-w-[80%] hover:bg-gray-50">
                    <p class="handwriting">Saya ingin mencari buku</p>
                </div>
                
                <div class="chat-suggestion bg-white/90 p-3 rounded-lg border-2 border-black max-w-[80%] hover:bg-gray-50">
                    <p class="handwriting">apa saja fasilitas di perpustakaan ini?</p>
                </div>
            </div>
            <?php endif; ?>
            
            <?php foreach ($chats as $chat): ?>
                <div class="flex items-end justify-end space-x-3 message-container">
                    <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                        <p class="text-right font-bold handwriting text-lg"><?= $user->nama; ?></p>
                        <p><?= $chat['user_message'] ?></p>
                        <div class="timestamp text-right"><?= $chat['timestamp'] ?></div>
                    </div>
                    <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                        <i class="fas fa-user"></i>
                    </div>
                </div>
                <div class="flex items-start space-x-3 message-container">
                    <div class="w-10 h-10 rounded-full border-2 border-black flex-shrink-0 flex items-center justify-center">
                        <i class="fas fa-book"></i>
                    </div>
                    <div class="bg-white p-4 rounded-lg border-2 border-black message-bubble max-w-[80%]">
                        <p class="font-bold handwriting text-lg">BookChat</p>
                        <p><?= $chat['bot_response'] ?></p>
                        <div class="timestamp"><?= $chat['timestamp'] ?></div>
                    </div>
                </div>
            <?php endforeach; ?>
        </div>
        
        <!-- Input form -->
        <div class="mt-5 border-t-2 border-gray-200 pt-4">
            <form id="chat-form" class="flex items-center space-x-3">
                <input type="text" id="message" name="message" placeholder="Tulis pesan Anda disini..." class="flex-grow p-3 border-2 border-black rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-300">
                <button type="submit" class="p-3 rounded-xl bg-white border-2 border-black hover:bg-gray-100 transition-colors h-12 w-12 flex items-center justify-center">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </form>
        </div>
    </div>

    <script src="<?php echo base_url('assets/js/main.js'); ?>"></script>
    <script>
        const baseUrl = "<?= base_url() ?>";
        const userName = <?= json_encode($user->nama); ?>;
        var activeController = "<?php echo $active_controller; ?>";                       
    </script>
</body>
</html>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perpus Bipa</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&family=Crimson+Text:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link href="<?php echo base_url(); ?>assets/css/main.css" rel="stylesheet">


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
                        <a href="<?php echo site_url($active_controller . '/clear'); ?>" onclick="return confirm('Yakin ingin menghapus semua riwayat chat Anda?')" class="flex items-center px-4 py-2 hover:bg-gray-100">
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

            <?php if (empty($chats)): ?>
                <!-- Chat suggestions only shown when no chat history -->
                <div class="flex flex-col space-y-3 mt-4 ml-12 items-end">
                    <h2>Contoh untuk memulai</h2>
                    <?php foreach ($suggestions as $text): ?>
                        <div class="chat-suggestion bg-white/90 p-3 rounded-lg border-2 border-black max-w-[80%] hover:bg-gray-50">
                            <p class="handwriting"><?= htmlspecialchars($text) ?></p>
                        </div>
                    <?php endforeach; ?>
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